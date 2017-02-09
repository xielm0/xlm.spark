package com.jd.szad.userlabel

import com.jd.szad.tools.Writer
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by xieliming on 2016/10/31.
 * 基于用户标签的推荐算法
 * 用户 x label , label x sku 的score用提升度
 * spark-shell --num-executors 40 --executor-memory 10g --executor-cores 5
 */
object app {

  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setAppName("user_label")
      .set("spark.akka.timeout", "1000")
      .set("spark.rpc.askTimeout", "500")

    val sc = new SparkContext(conf)

    //params
    val model_type = args(0)

    val sqlContext = new org.apache.spark.sql.hive.HiveContext(sc)

    if (model_type == "train") {
      sqlContext.sql("set spark.sql.shuffle.partitions = 400")
      val sql =
        """
          |select  uid,type,label,sku, 1 as rate
          |from app.app_szad_m_dyrec_userlabel_train
          |where type in('browse_top20sku','browse_top20brand','browse_top20cate')
        """.stripMargin
      val df = sqlContext.sql(sql)

      // type,label,sku
      val df1 = df.groupBy("type", "label", "sku").agg(sum("rate") as "sku_uv").filter(col("sku_uv") > 10).cache()
      val df2 = df1.groupBy("type", "label").agg(sum("sku_uv") as "label_uv")
      val df3 = df1.groupBy("type").agg(sum("sku_uv") as "type_uv")
      val df4 = df1.groupBy("type", "sku").agg(sum("sku_uv") as "sku_uv").filter(col("sku_uv") > 10)
      println("df3.count is " + df3.count())
      //      val test= df2.filter(col("type")=== "browse_top20sku"  and col("label")=== "1283994")

      // prob
      val sku_label_rate = df1.join(df2, Seq("type", "label")).selectExpr("type", "label", "sku", "round(sku_uv/label_uv,8) as sku_label_rate")
      val sku_type_rate = df4.join(df3, Seq("type")).selectExpr("type", "sku", "round(sku_uv/type_uv,8) as sku_type_rate")

      // lift
      val lift = sku_label_rate.join(sku_type_rate, Seq("type", "sku")).selectExpr("type", "label", "sku", "round(sku_label_rate/sku_type_rate,4) as lift")
      val lift_1 = lift.where(col("lift") > 10).where("label <> String(sku) " )

      //save
      //      val res = lift_1.rdd.map(t=>t(0) + "\t" + t(1) + "\t" + t(2) + "\t" + t(3))
      //      Writer.write_table(res,"app.app_szad_m_dyrec_userlabel_model","lzo")
      sqlContext.sql("use app")
      lift_1.registerTempTable("res_table")
      sqlContext.sql("set mapreduce.output.fileoutputformat.compress=true")
      sqlContext.sql("set hive.exec.compress.output=true")
      sqlContext.sql("set mapred.output.compression.codec=com.hadoop.compression.lzo.LzopCodec")

      sqlContext.sql("insert overwrite table app.app_szad_m_dyrec_userlabel_model select type,label,sku,lift from res_table")

    } else if (model_type == "predict") {
      /* 用户 x label , label x sku ,这里，一个用户不超过100个标签，一个label对应不超过100个sku。否则计算困难。
       * 最终保存的结果，每个用户保存top 100
       * type_id=join,则通过spark sql来实现，否则通过 矩阵相乘。
       * Unable to acquire 33554432 bytes of memory
       *  */

      val sql =
        s"""
           |select uid,concat(type,'_',label) as label, rate
           |from app.app_szad_m_dyrec_userlabel_apply
           |where user_type=1
           |and length(uid)>20
           |and rn<=20
        """.stripMargin
      val df_user_label = sqlContext.sql(sql)

      val sql2 =
        s"""
           |insert overwrite table app.app_szad_m_dyRec_userlabel_model_2
           |select type ,label,sku,score
           | from (select sku,type,label,score,row_number() over(partition by type,label order by score desc ) rn
           |         from app.app_szad_m_dyRec_userlabel_model
           |        where label <> String(sku) )t
           | where rn <=20
        """.stripMargin

      sqlContext.sql(sql2)

      val sql3 =
        s"""
           |select concat(type,'_',label) as label,sku,score from app_szad_m_dyRec_userlabel_model_2
        """.stripMargin
      val df_label_sku = sqlContext.sql(sql3)


        //   矩阵相乘
        //   spark1.5.2跑这个大数据量的sql有bug,必须1.6以上的版本
        //   df的join会被解析成sql
        //   有数据倾斜

        sqlContext.sql("set spark.sql.shuffle.partitions = 2000")

        //df join
        val df1 = df_user_label.join(df_label_sku, "label").selectExpr("uid", "sku", "rate*score as score")
        val df2 = df1.groupBy("uid", "sku").agg(round(sum("score"), 4) as "score")

        //top 100
        val w = Window.partitionBy("uid").orderBy(desc("score"))
        val df4 = df2.select(col("uid"), col("sku"), col("score"), rowNumber().over(w).alias("rn")).where("rn<=100")

        //save
        df4.printSchema()
        val res = df4.rdd.map(t => t.getAs("uid").toString + "\t" + t.getAs("sku") + "\t" + t.getAs("score") + "\t" + t.getAs("rn") )
//        println("total count = " + res.count())
        Writer.write_table( res ,"app.db/app_szad_m_dyrec_userlabel_predict_res/user_type=1","lzo")

//        //join
//        val user_label = df_user_label.rdd.map(t=>(t(1),(t(0),t(2).asInstanceOf[Long]))).repartition(2000)
//        val label_sku =  df_label_sku.rdd.map(t=>(t(0),(t(1),t(2).asInstanceOf[Double])))
//        val bc_b =sc.broadcast(label_sku)
//        val rdd1 = user_label.join(bc_b.value)
//          .map{case (label,((user,rate),(sku,score))) => ((user,sku),rate*score)}
//          .reduceByKey(_+_).map(t=> (t._1,math.round(1000*t._2)/1000.0))
//
//        //top
//        val rdd2 = rdd1.map{ case((uid,sku),score)=>(uid,(sku,score))}
//          .groupByKey()
//          .flatMap {case (uid, b) =>
//            val topK = b.toArray.sortWith { (a, b) => a._2 > b._2 }.take(100)
//            topK.zipWithIndex.map(t => (uid, t._1._1, t._1._2, t._2)) //t._2 = rn
//          }

        //save
//        val res = rdd2.map(t=>t._1 + "\t" + t._2 + "\t" + t._3 + "\t" + t._4)
//        Writer.write_table( res ,"app.db/app_szad_m_dyrec_userlabel_predict_res/user_type=1","lzo")

    }



  }


}
