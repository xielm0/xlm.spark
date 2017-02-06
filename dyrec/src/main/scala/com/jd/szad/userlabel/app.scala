package com.jd.szad.userlabel

import com.jd.szad.tools.Writer
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.sql.functions._
import org.apache.spark.{SparkConf, SparkContext}
/**
 * Created by xieliming on 2016/10/31.
 * 基于用户标签的推荐算法
 * 用户 x label , label x sku 的score用提升度
 * spark-shell --num-executors 50 --executor-memory 8g --executor-cores 4
 */
object app {

  def main(args:Array[String]) {
    val conf = new SparkConf()
      .setAppName("user_label")
      .set("spark.akka.timeout", "1000")
      .set("spark.rpc.askTimeout", "500")
      .set("spark.storage.memoryFraction","0.1")

    val sc = new SparkContext(conf)

    //params
    val model_type =args(0)

    val sqlContext = new org.apache.spark.sql.hive.HiveContext(sc)

    if (model_type =="train") {
      sqlContext.sql("set spark.sql.shuffle.partitions = 400")
      val sql =
        """
          |select  uid,type,label,sku, 1 as rate
          |from app.app_szad_m_dyrec_userlabel_train_t
          |where type in('browse_top20sku','browse_top20brand','browse_top20_cate')
        """.stripMargin
      val df = sqlContext.sql(sql)

      // type,label,sku
      val df1 = df.groupBy("type","label","sku").agg(sum("rate") as "sku_uv").filter(col("sku_uv")>10).cache()
      val df2 = df1.groupBy("type","label").agg(sum("sku_uv") as "label_uv")
      val df3 = df1.groupBy("type").agg(sum("sku_uv") as "type_uv")
      val df4 = df1.groupBy("type","sku").agg(sum("sku_uv") as "sku_uv" ).filter(col("sku_uv")>10)
      println("df3.count is " + df3.count())
//      val test= df2.filter(col("type")=== "browse_top20sku"  and col("label")=== "1283994")

      // prob
      val sku_label_rate = df1.join(df2, Seq("type","label")).selectExpr("type","label", "sku", "round(sku_uv/label_uv,8) as sku_label_rate")
      val sku_type_rate = df4.join(df3, Seq("type")).selectExpr("type", "sku", "round(sku_uv/type_uv,8) as sku_type_rate")

      // lift
      val lift = sku_label_rate.join(sku_type_rate,Seq("type", "sku")).selectExpr("type", "label", "sku", "round(sku_label_rate/sku_type_rate,4) as lift")
      val lift_1 = lift.where(col("lift")>2 and col("label") !== col("sku").toString())

      //save
      sqlContext.sql("use app")
      lift_1.registerTempTable("res_table")
      sqlContext.sql("set mapreduce.output.fileoutputformat.compress=true")
      sqlContext.sql("set hive.exec.compress.output=true")
      sqlContext.sql("set mapred.output.compression.codec=com.hadoop.compression.lzo.LzopCodec")

      sqlContext.sql("insert overwrite table app.app_szad_m_dyrec_userlabel_model select type,label,sku,lift from res_table")

     }else if (model_type =="sql_predict") {
      /* 用户 x label , label x sku ,这里，一个用户不超过100个标签，一个label对应不超过100个sku。否则计算困难。
       * 最终保存的结果，每个用户保存top 100
       * spark1.5.2跑这个大数据量的sql有bug,必须1.6以上的版本
       *  */
      val v_day = args(1)
      sqlContext.sql("set spark.sql.shuffle.partitions = 1000")
      sqlContext.sql("set mapreduce.output.fileoutputformat.compress=true")
      sqlContext.sql("set hive.exec.compress.output=true")
      sqlContext.sql("set mapred.output.compression.codec=com.hadoop.compression.lzo.LzopCodec")

      val sql1=
        s"""
          |insert overwrite table app.app_szad_m_dyrec_userlabel_apply partition (user_type=1)
          |select uid,'browse_top20sku' as type,sku as label,1 as rate ,rn
          |  from app_szad_m_dyrec_user_top100_sku
          | where user_type=1 and action_type=1 and dt='${v_day}'
          |   and rn<=20
          |union all
          |select uid,'browse_top20cate' as type,3rd_cate_id as label,count(1) as rate ,rn
          |  from app_szad_m_dyrec_user_top100_sku
          | where user_type=1 and action_type=1 and dt='${v_day}'
          |   and rn<=20
          |group by uid,3rd_cate_id,rn
        """.stripMargin
//      sqlContext.sql(sql1)


      val sql2=
        """
          |insert overwrite table app.app_szad_m_dyrec_userlabel_predict_res partition (user_type=1)
          |select uid ,sku ,score,rn
          |  from( select uid ,sku ,score ,row_number() over(partition by uid order by score desc) rn
          |          from(select uid,sku,sum(round(rate*score,1)) as score
          |                from (select * from app_szad_m_dyrec_userlabel_apply where user_type=1 and rn <=20) a
          |                join (select type,label,sku,score
          |                       from (select type,label,sku,score,row_number() over(partition by type,label order by score desc ) rn
          |                               from app.app_szad_m_dyRec_userlabel_model )t
          |                       where rn <=20 )b
          |                 on (a.type=b.type and a.label=b.label)
          |              group by uid,sku
          |                )t1
          |       )t2
          | where rn <=100
        """.stripMargin
      sqlContext.sql(sql2)

    }else if (model_type =="predict") {
      /*
      * 转换成点矩阵，然后join,或者说矩阵相乘
      * */
      val v_day = args(1)

      val sql=
        s"""
          |select uid,concat(type,'_',label) as label, rate
          |from(
          |select uid,'browse_top20sku' as type,sku as label,1 as rate
          |  from app_szad_m_dyrec_user_top100_sku
          | where user_type=1 and action_type=1 and dt='${v_day}'
          |   and rn<=20
          |union all
          |select uid,'browse_top20cate' as type,3rd_cate_id as label,count(1) as rate
          |  from app_szad_m_dyrec_user_top100_sku
          | where user_type=1 and action_type=1 and dt='${v_day}'
          |   and rn<=20
          |group by uid,3rd_cate_id,rn
          |)t
        """.stripMargin
      val user_label = sqlContext.sql(sql).rdd.map(t=>(t(0),t(1),t(2).asInstanceOf[Int]))

      val sql2 =
        s"""
          |select concat(type,'_',label) as label,sku,score
          | from (select sku,type,label,score,row_number() over(partition by type,label order by score desc ) rn
          |         from app.app_szad_m_dyRec_userlabel_model)t
          | where rn <=20
        """.stripMargin
      val label_sku =sqlContext.sql(sql2).rdd.map(t=>(t(0),t(1),t(2).asInstanceOf[Double])).cache()

      //labels map
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
      val mat_A  = new CoordinateMatrix( A ).toBlockMatrix(10240,10240)
      val mat_B  = new CoordinateMatrix( B ).toBlockMatrix(10240,10240)
      val S = mat_A.multiply(mat_B)

      println("numRowBlocks= " + mat_A.numRowBlocks + ",numColBlocks"+mat_B.numColBlocks)  //numColBlocks=180
      println("A.blocks.partitions.length"+mat_A.blocks.partitions.length + "B.blocks.partitions.length"+mat_B.blocks.partitions.length)
      println("first block size = " + S.blocks.map(t=>t._2.toString()))

      // top 100
      val top100 =S.toCoordinateMatrix().entries.map(t =>(t.i,(t.j,t.value))).groupByKey().flatMap{
        case( a, b)=>  //b=Interable[(item2,score)]
          val topk= b.toArray.sortWith{ (b1,b2) => b1._2 > b2._2 }.take(100)
//          topk.map{ t => (a,t._1,t._2) }
          topk.zipWithIndex.map{ t => (a,t._1._1,t._1._2,t._2) }
      }
      val res = top100.map(t=>t._1 + "\t" + t._2 + "\t" + t._3 + "\t" + t._4)

      //save
      Writer.write_table( res ,"app.db/app_szad_m_dyrec_userlabel_predict_res/user_type=1","lzo")

    }
  }


}
