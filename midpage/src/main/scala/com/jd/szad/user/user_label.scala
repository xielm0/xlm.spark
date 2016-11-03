package com.jd.szad.user

import com.jd.szad.tools.Writer
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
/**
 * Created by xieliming on 2016/10/31.
 * 基于用户标签的推荐算法
 * 用户 x label , label x sku 的score用提升度
 * spark-shell --num-executors 10 --executor-memory 8g --executor-cores 4
 */
object user_label {
  def main(args:Array[String]) {
    val conf = new SparkConf()
      .setAppName("user_label")
      .set("spark.akka.timeout", "1000")
      .set("spark.rpc.askTimeout", "500")
      .set("spark.storage.memoryFraction","0.1")

    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.hive.HiveContext(sc)
    import org.apache.spark.sql.functions._

    val model_type =args(0)

    if (model_type =="train") {
      val sql =
        """
        |select type,feature,sku,cnt
        |from(
        |select  type ,feature,sku,sum(1) as cnt
        |from app.app_szad_m_midpage_jd_label_train
        |group by type,feature,sku)t
        |where cnt>=10
      """.
          stripMargin
      val df = sqlContext.sql(sql).cache()
      df.filter(df("type")===2).count()
      //sku
      val df_sku = df.groupBy("type", "sku").agg(sum("cnt") as "sku_cnt")
      val df_sum = df.groupBy("type").agg(sum("cnt") as "sum_cnt")
      val sku_prob = df_sku.join(df_sum, "type").selectExpr("type", "sku", "round(sku_cnt/sum_cnt,8) as sku_prob")
      //feature
      val df_f = df.groupBy("type", "feature").agg(sum("cnt") as "f_cnt")
      val f_prob = df.join(df_f,Seq("type", "feature")).selectExpr("type", "feature", "sku", "cnt", "f_cnt", "round(cnt/f_cnt,6) as f_prob")
      val t1 = f_prob.join(sku_prob,Seq("type", "sku")).selectExpr("type", "feature", "sku", "round(f_prob/sku_prob,4) as boost_prob")
      val boost_prob = t1.where(col("boost_prob")>1)
      // require boost_prob>1,so log(2,boost_prob)>0
      // 性别、年龄的提升度不会很大，但是对于类目这种，则value is large
      //  test
      //  df.where(col("feature")===9729 && col("sku")==="10613996786").show()

      //save
      //boost_prob.write.save("app.db/app_szad_m_midpage_jd_label_model")
      sqlContext.sql("use app")
      boost_prob.registerTempTable("res_table")
      sqlContext.sql("insert overwrite table app.app_szad_m_midpage_jd_label_model select type,feature,sku,boost_prob from res_table")

     }else if (model_type =="sql_predict") {
      /* 面临的几个问题
       * 类目偏好和购物性别对应的sku验证的数据倾斜；解决办法：分开
       *  */
      sqlContext.sql("set spark.sql.shuffle.partitions = 1000")
      val sql=
        """
          |insert overwrite table app.app_szad_m_midpage_jd_label_res
          |select uid,sku,sum(round(rate*score,6)) as score
          |from (select uid,type,feature, 1 as rate
          |        from app.app_szad_m_midpage_jd_label_train
          |        where type=3
          |       group by  uid,type,feature )a
          |join (select sku,type,feature,score
          |        from app.app_szad_m_midpage_jd_label_model
          |         where type=3)b
          |on(a.type=b.type and a.feature=b.feature)
          |group by uid,sku
        """.stripMargin
      sqlContext.sql(sql)

    }else if (model_type =="predict") {

      val sql2=
        """
          |select uid ,concat(type,'_',feature) as feature,1 as rate
          |from app.app_szad_m_midpage_jd_label_train
          |where length(uid)<=40
          |group by uid,concat(type,'_',feature)
        """.stripMargin
      val user_label = sqlContext.sql(sql2).rdd.map(t=>(t(0).asInstanceOf[String],t(1).asInstanceOf[String],t(2).asInstanceOf[Int]))

      val label_sku =sqlContext.sql("select sku,concat(type,'_',feature) as feature,score from app.app_szad_m_midpage_jd_label_model").rdd
      .map(t=>(t(0).asInstanceOf[String],t(1).asInstanceOf[String],t(2).asInstanceOf[Double]))

      val res = predict(user_label,label_sku).map(t=> t._1 + "\t" + t._2)

      //save
      Writer.write_table(res,"app.db/app_szad_m_midpage_jd_label_res")

    }
  }

  /*
  * user_label : uid,feature,rate
  * label_sku :  feature,sku,score
  * */

  def predict(user_label:RDD[(String,String,Int)],
              label_sku:RDD[(String,String,Double)]): RDD[(Long,String)] ={
    //features map
    label_sku.cache()
    val labels = label_sku.map(_._1).distinct().zipWithIndex().map(t=>(t._1,t._2)).collectAsMap()
    val skus     = label_sku.map(_._2).distinct().zipWithIndex().map(t=>(t._1,t._2)).collectAsMap()
    val users    = user_label.map(_._1).distinct().zipWithIndex().map(t=>(t._1,t._2)).collectAsMap()

    val l_size = labels.size
    //
    val A = user_label.map{ t =>
      val user = users( t._1)
      val label = labels( t._2)
      MatrixEntry(user,label,t._3)
    }

    val B = label_sku.map{ t =>
      val label = labels( t._1)
      val sku = skus( t._2)
      MatrixEntry(label,sku,t._3)
    }

    //compute
    // block matrix *  block matrix
    val mat_A  = new CoordinateMatrix( A ).toBlockMatrix(2048,2048)
    val mat_B  = new CoordinateMatrix( B ).toBlockMatrix(2048,2048)

    val S = mat_A.multiply(mat_B)
    println("first block size = " + S.blocks.first().toString())

    val res = S.toIndexedRowMatrix().rows.map(t=>(t.index,t.vector.toSparse.toArray.mkString(":")))
    res

  }


}
