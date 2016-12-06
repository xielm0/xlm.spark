package com.jd.szad.user

import com.jd.szad.tools.Writer
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
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
      sqlContext.sql("set spark.sql.shuffle.partitions = 2000")

      val sql=
        """
          |insert overwrite table app.app_szad_m_midpage_jd_label_res
          |select uid ,sku ,score
          |  from( select uid ,sku ,score ,row_number() over(partition by uid order by score desc) rn
          |          from(select uid,sku,sum(round(rate*score,1)) as score
          |                from (select uid,type,feature, 1 as rate,count(1)
          |                        from app.app_szad_m_midpage_jd_label_train
          |                       group by  uid,type,feature )a
          |                join (select sku,type,feature,score
          |                       from (select sku,type,feature,score,row_number() over(partition by type,feature order by score desc ) rn
          |                               from app.app_szad_m_midpage_jd_label_model)t
          |                              where rn <=200 )b
          |                 on (a.type=b.type and a.feature=b.feature)
          |                join (select uid as valid_user from(
          |                         select uid,count(distinct feature) cnt from app.app_szad_m_midpage_jd_label_train group by uid)t
          |                      where cnt <=100 )c
          |                  on a.uid = c.valid_user
          |              group by uid,sku
          |                )t1
          |       )t2
          | where rn <=200
        """.stripMargin
      sqlContext.sql(sql)

    }else if (model_type =="predict") {

      val sql=
        """
          |select uid ,concat(type,'_',feature) as feature,1 as rate
          |from app.app_szad_m_midpage_jd_label_train
          |where length(uid)<=40
          |group by uid,concat(type,'_',feature)
        """.stripMargin
      val user_label = sqlContext.sql(sql).rdd.map(t=>(t(0),t(1),t(2).asInstanceOf[Int]))

      val sql2 =
        """
          |select feature,sku,score
          |from(
          |select concat(type,'_',feature) as feature,sku,score,row_number() over(partition by type,feature order by score desc ) rn
          |from app.app_szad_m_midpage_jd_label_model
          |)t where rn<=200
        """.stripMargin
      val label_sku =sqlContext.sql(sql2).rdd.map(t=>(t(0),t(1),t(2).asInstanceOf[Double])).cache()

      //features map
      val labels = sc.broadcast(label_sku.map(_._1).distinct().zipWithIndex().collectAsMap() )
      val skus   = sc.broadcast(label_sku.map(_._2).distinct().zipWithIndex().collectAsMap() )
      val users  = user_label.map(_._1).distinct().zipWithIndex()

      //
      val A = user_label.map(t=>(t._1,(t._2,t._3))).join(users).
        map{ t =>
          val user = t._2._2
          if (labels.value.contains(t._2._1._1) ){
              val label = labels.value( t._2._1._1)
              MatrixEntry(user,label,t._2._1._2)
          }else{MatrixEntry(-1,-1,0)}
        }.filter(t=>t.i>=0)

      val B = label_sku.map{ t =>
        val label = labels.value( t._1)
        val sku = skus.value( t._2)
        MatrixEntry(label,sku,t._3)
      }

      //compute
      // block matrix *  block matrix
      val mat_A  = new CoordinateMatrix( A ).toBlockMatrix(1024,1024)
      val mat_B  = new CoordinateMatrix( B ).toBlockMatrix(1024,1024)
      val S = mat_A.multiply(mat_B)

      println("numRowBlocks= " + mat_A.numRowBlocks + ",numColBlocks"+mat_B.numColBlocks)  //numColBlocks=180
      println("A.blocks.partitions.length"+mat_A.blocks.partitions.length + "B.blocks.partitions.length"+mat_B.blocks.partitions.length)
      println("first block size = " + S.blocks.map(t=>t._2.toString()))

      val res =S.toCoordinateMatrix().entries.map(t=>t.i + "\t" + t.j + "\t" + t.value)

      //save
      Writer.write_table( res ,"app.db/app_szad_m_midpage_jd_label_res","lzo")

    }
  }


}