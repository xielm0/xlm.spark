package com.jd.szad.userlabel

import com.jd.szad.tools.Writer
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel
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
          |where type in('short_sku2browse','short_sku2click','long_sku2browse','long_sku2click')
          |group by uid,type,label,sku
        """.stripMargin
      val df = sqlContext.sql(sql).persist(StorageLevel.MEMORY_AND_DISK)


      // type,label,sku
      val df1 = df.groupBy("type", "label", "sku").agg(sum("rate") as "sku_uv").filter(col("sku_uv") > 10)
      val df2 = df.groupBy("type", "label").agg(countDistinct("uid") as "label_uv").filter(col("label_uv") > 10)

      val label_sku =  df1.join(df2, Seq("type", "label")).selectExpr("type", "label", "sku", "round(sku_uv/log(1+label_uv),4) as score")

      //save
//      val res = label_sku.rdd.map(t=>t.getAs("type").toString + "\t" + t.getAs("label") + "\t" + t.getAs("sku") + "\t" + t.getAs("score"))
//      Writer.write_table(res,"app.app_szad_m_dyrec_userlabel_model","lzo")
      sqlContext.sql("use app")
      label_sku.registerTempTable("res_table")
      sqlContext.sql("set mapreduce.output.fileoutputformat.compress=true")
      sqlContext.sql("set hive.exec.compress.output=true")
      sqlContext.sql("set mapred.output.compression.codec=com.hadoop.compression.lzo.LzopCodec")

      sqlContext.sql("insert overwrite table app.app_szad_m_dyrec_userlabel_model select type,label,sku,score from res_table where label <> String(sku)")

    } else if (model_type =="predict") {
      // 无数据倾斜时
      // spark1.5.2跑这个大数据量的sql有bug,必须1.6以上的版本

      val cond1 = args(1).toString
      val cond2 = args(2).toString
      val output_path = args(3)

      val sql1 =
        s"""
           |select uid,label, rate
           |from app.app_szad_m_dyrec_userlabel_apply
           |where user_type=1
           |and length(uid)>20
           |and  type='${cond1}'
        """.stripMargin
      val df_user_label = sqlContext.sql(sql1)

      val sql2 =
        s"""
           |select label,sku,score
           |  from (select sku,type,label,score,row_number() over(partition by type,label order by score desc ) rn
           |         from app.app_szad_m_dyrec_userlabel_model
           |        where type='${cond2}'
           |          and label <> String(sku)  )t
           |where rn<=10
        """.stripMargin
      val df_label_sku = sqlContext.sql(sql2)

      // join
      val partitions=1000
//      val k =50
      sqlContext.sql(s"set spark.sql.shuffle.partitions = ${partitions}")

      //df join
      val df1 = df_user_label.join(df_label_sku, "label").selectExpr("uid", "sku", "rate*score as score")
      val df2 = df1.groupBy("uid", "sku").agg(round(sum("score"), 4) as "score")

      //top 100
      val k =20
      val w = Window.partitionBy("uid").orderBy(desc("score"))
      val df4 = df2.select(col("uid"), col("sku"), col("score"), rowNumber().over(w).alias("rn")).where(s"rn<=${k}")

      //save
      val res = df4.rdd.map(t => t.getAs("uid").toString + "\t" + t.getAs("sku") + "\t" + t.getAs("score") + "\t" + t.getAs("rn") )
      Writer.write_table( res ,output_path,"lzo")

    }else if (model_type =="predict2") {
      // 数据倾斜时

      val cond1 = args(1).toString
      val cond2 = args(2).toString
      val output_path = args(3)
      val partitions =500

      val sql1 =
        s"""
           |select uid,label, rate
           |from app.app_szad_m_dyrec_userlabel_apply
           |where user_type=1
           |and length(uid)>20
           |and  type='${cond1}'
        """.stripMargin
      val df_user_label = sqlContext.sql(sql1)

      val sql2 =
        s"""
           |select label,sku,score
           |  from (select sku,type,label,score,row_number() over(partition by type,label order by score desc ) rn
           |         from app.app_szad_m_dyrec_userlabel_model
           |        where type='${cond2}'
           |          and label <> String(sku)  )t
           |where rn<=10
        """.stripMargin
      val df_label_sku = sqlContext.sql(sql2)

      val t2 = df_label_sku.rdd.map(t=>(t.getAs("label").toString,(t.getAs("sku").toString,t.getAs("score").asInstanceOf[Double]))).groupByKey().collectAsMap()
      val bc_t2 = sc.broadcast(t2)
      val t1 = df_user_label.rdd.map(t=>(t.getAs("uid").toString,t.getAs("label").toString,t.getAs("rate").asInstanceOf[Int])).repartition(partitions)

      val rdd_join =t1.mapPartitions{iter =>
        for{(uid,label,rate) <- iter
            if (bc_t2.value.contains(label))
            skus= bc_t2.value.get(label)
            sku <- skus.get
        }  yield (uid, sku._1, math.round(sku._2 * rate *10000)/10000.0 )
      }

      //top 100
      val k =20
      val top100 = rdd_join.map(t=> (t._1,(t._2,t._3))).groupByKey().flatMap{
        case( a, b)=>  //b=Interable[(sku,score)]
          val bb = b.toBuffer
          val topk =bb.sortBy(_._2).reverse
          if (topk.length >k ) topk.remove( k, topk.length-k)
          topk.zipWithIndex.map{case((sku,score),rn) => (a, sku,score,rn) }
      }

      //save
      val res = top100.map(t=>t._1 + "\t" + t._2.toString + "\t" + t._3.toString + "\t" + t._4 )
      Writer.write_table( res ,output_path,"lzo")

    }

  }


}
