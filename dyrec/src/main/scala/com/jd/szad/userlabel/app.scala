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
    val sqlContext = new org.apache.spark.sql.hive.HiveContext(sc)

    //params
    val model_type = args(0)

    if (model_type == "train") {
      sqlContext.sql("set spark.sql.shuffle.partitions = 1000")
      val sql =
        """
          |select  uid,type,label,sku, 1 as rate
          |from app.app_szad_m_dyrec_userlabel_train
          |where type in('browse_top20sku','browse_top20brand','browse_top20cate','7_click_top20cate','7_click_top20brand','7_click_top20sku')
        """.stripMargin
      val df = sqlContext.sql(sql)

      // type,label,sku
      val df1 = df.groupBy("type", "label", "sku").agg(sum("rate") as "sku_uv").filter(col("sku_uv") > 10).cache()
      val df2 = df1.groupBy("type", "label").agg(sum("sku_uv") as "label_uv")
      val df3 = df1.groupBy("type").agg(sum("sku_uv") as "type_uv")
      val df4 = df1.groupBy("type", "sku").agg(sum("sku_uv") as "sku_uv").filter(col("sku_uv") > 10)
      println("df3.count is " + df3.count())
      // val test= df2.filter(col("type")=== "browse_top20sku"  and col("label")=== "1283994")

      // prob
      val sku_label_rate = df1.join(df2, Seq("type", "label")).selectExpr("type", "label", "sku", "round(sku_uv/label_uv,8) as sku_label_rate")
      val sku_type_rate = df4.join(df3, Seq("type")).selectExpr("type", "sku", "round(sku_uv/type_uv,8) as sku_type_rate")

      // lift
      val lift = sku_label_rate.join(sku_type_rate, Seq("type", "sku")).selectExpr("type", "label", "sku", "round(sku_label_rate/sku_type_rate,4) as lift")
      val lift_1 = lift.where(col("lift") > 5).where("label <> String(sku) " )

      //save
      //      val res = lift_1.rdd.map(t=>t(0) + "\t" + t(1) + "\t" + t(2) + "\t" + t(3))
      //      Writer.write_table(res,"app.app_szad_m_dyrec_userlabel_model","lzo")
      sqlContext.sql("use app")
      lift_1.registerTempTable("res_table")
      sqlContext.sql("set mapreduce.output.fileoutputformat.compress=true")
      sqlContext.sql("set hive.exec.compress.output=true")
      sqlContext.sql("set mapred.output.compression.codec=com.hadoop.compression.lzo.LzopCodec")

      sqlContext.sql("insert overwrite table app.app_szad_m_dyrec_userlabel_model select type,label,sku,lift from res_table where label <> String(sku)")

    } else if (model_type =="predict") {
      // 无数据倾斜时
      // spark1.5.2跑这个大数据量的sql有bug,必须1.6以上的版本

      val cond1 = args(1).toString
      val cond2 = args(2).toString
      val output_path = args(3)

      val sql =
        s"""
           |select uid,label, rate
           |from app.app_szad_m_dyrec_userlabel_apply
           |where user_type=1
           |and length(uid)>20
           |and rn<=20
           |and  type='${cond1}'
        """.stripMargin
      val df_user_label = sqlContext.sql(sql)

      val sql2 =
        s"""
           |select label,sku,score
           |from app.app_szad_m_dyRec_userlabel_model_2
           |where type='${cond2}'
        """.stripMargin
      val df_label_sku = sqlContext.sql(sql2)

      // join
      val partitions=2000
//      val k =50
      sqlContext.sql(s"set spark.sql.shuffle.partitions = ${partitions}")

      //df join
      val df1 = df_user_label.join(df_label_sku, "label").selectExpr("uid", "sku", "rate*score as score")
      val df2 = df1.groupBy("uid", "sku").agg(round(sum("score"), 4) as "score")

      //top 100
      val k =50
      val w = Window.partitionBy("uid").orderBy(desc("score"))
      val df4 = df2.select(col("uid"), col("sku"), col("score"), rowNumber().over(w).alias("rn")).where(s"rn<=${k}")

      //save
      val res = df4.rdd.map(t => t.getAs("uid").toString + "\t" + t.getAs("sku") + "\t" + t.getAs("score") + "\t" + t.getAs("rn") )
      Writer.write_table( res ,output_path,"lzo")

    }else if (model_type =="predict2") {
      // 数据倾斜时，

      val cond1 = args(1).toString
      val cond2 = args(2).toString
      val output_path = args(3)
      val partitions =2000

      val sql =
        s"""
           |select uid,label, rate
           |from app.app_szad_m_dyrec_userlabel_apply
           |where user_type=1
           |and length(uid)>20
           |and rn<=20
           |and  type='${cond1}'
        """.stripMargin
      val df_user_label = sqlContext.sql(sql)

      val sql2 =
        s"""
           |select label,sku,score
           |from app.app_szad_m_dyRec_userlabel_model_2
           |where type='${cond2}'
        """.stripMargin
      val df_label_sku = sqlContext.sql(sql2)

      /*
      val sql3=
      s"""
        |select /*+mapjoin(b)*/ uid,sku,rate*score as score
        |from(select uid,label, rate
        |      from app.app_szad_m_dyrec_userlabel_apply
        |      where user_type=1
        |      and length(uid)>20
        |      and rn<=20
        |      and  type='${cond1}')a
        |join (select * from app.app_szad_m_dyRec_userlabel_model_2 where type='${cond2}')b
        |on a.label =b.label
      """.stripMargin
      val df_user_sku = sqlContext.sql(sql3)

      sqlContext.sql(s"set spark.sql.shuffle.partitions = ${partitions}")
      //top 100
      val w = Window.partitionBy("uid").orderBy(desc("score"))
      val df4 = df_user_sku.select(col("uid"), col("sku"), col("score"), rowNumber().over(w).alias("rn")).where(s"rn<=${k}")
      val res = df4.rdd.map(t => t.getAs("uid").toString + "\t" + t.getAs("sku") + "\t" + t.getAs("score") + "\t" + t.getAs("rn") )
*/

      val t2 = df_label_sku.rdd.map(t=>(t.getAs("label").toString,(t.getAs("sku").toString,t.getAs("score").asInstanceOf[Double]))).groupByKey().collectAsMap()
      val bc_t2 = sc.broadcast(t2)
      val t1 = df_user_label.rdd.map(t=>(t.getAs("uid").toString,t.getAs("label").toString,t.getAs("rate").asInstanceOf[Int]))

      val rdd_join =t1.mapPartitions{iter =>
        for{(uid,label,rate) <- iter
            if (bc_t2.value.contains(label))
            skus= bc_t2.value.get(label)
            sku <- skus.get
        }  yield (uid, sku._1, math.round(sku._2 * rate *10000)/10000.0 )
      }

      //top 100
      val k =50
      val top100 = rdd_join.map(t=> (t._1,(t._2,t._3))).groupByKey().flatMap{
        case( a, b)=>  //b=Interable[(sku,score)]
          val bb = b.toBuffer
          val topk =bb.sortBy(_._2).reverse
          if (topk.length >k ) topk.remove( k, topk.length-k)
          topk.zipWithIndex.map{case((sku,score),rn) => (a, sku,score,rn) }
      }

      //save
      val res = top100.map(t=>t._1 + "\t" + t._2 + "\t" + t._3.toString + "\t" + t._4 + "\t")
      Writer.write_table( res ,output_path,"lzo")

    }

  }


}
