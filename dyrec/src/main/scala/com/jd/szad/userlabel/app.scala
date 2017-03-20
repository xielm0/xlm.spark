package com.jd.szad.userlabel

import com.jd.szad.tools.Writer
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
          |where type in('sku','sku_feature'  )
          |group by uid,type,label,sku
        """.stripMargin
      val df = sqlContext.sql(sql).persist(StorageLevel.MEMORY_AND_DISK)

      // type,label,sku
      val minSupport =10
      val df1 = df.groupBy("type", "label", "sku").agg(sum("rate") as "sku_uv").filter(col("sku_uv") > minSupport)
//      val df2 = df.groupBy("type", "label").agg(countDistinct("uid") as "label_uv").filter(col("label_uv") > minSupport)
      val df3 = df.groupBy("type","sku").agg(countDistinct("uid") as "sku_all_uv").filter(col("sku_all_uv") > minSupport)

      val label_sku =  df1.join(df3, Seq("type", "sku")).selectExpr("type", "label", "sku", "round(sku_uv/log(1+sku_all_uv),4) as score")

      //save
//      val res = label_sku.rdd.map(t=>t.getAs("type").toString + "\t" + t.getAs("label") + "\t" + t.getAs("sku") + "\t" + t.getAs("score"))
//      Writer.write_table(res,"app.app_szad_m_dyrec_userlabel_model","lzo")
      sqlContext.sql("use app")
      label_sku.registerTempTable("res_table")
      sqlContext.sql("set mapreduce.output.fileoutputformat.compress=true")
      sqlContext.sql("set hive.exec.compress.output=true")
      sqlContext.sql("set mapred.output.compression.codec=com.hadoop.compression.lzo.LzopCodec")

      sqlContext.sql("insert overwrite table app.app_szad_m_dyrec_userlabel_model select type,label,sku,score from res_table where label <> String(sku)")

    } else if (model_type =="predict2") {
      // 数据倾斜时

      val cond1 = args(1).toString
      val cond2 = args(2).toString
      val output_path = args(3)
      val partitions =500
      sqlContext.sql(s"set spark.sql.shuffle.partitions = ${partitions}")

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
           |where rn<=20
        """.stripMargin
      val df_label_sku = sqlContext.sql(sql2)

      val t2 = df_label_sku.rdd.map(t=>(t.getAs("label").toString,(t.getAs("sku").toString,t.getAs("score").asInstanceOf[Double]))).groupByKey().collectAsMap()
      val bc_t2 = sc.broadcast(t2)

      //历史上的label分布不均匀，且数量庞大。
      val df1 = df_user_label.groupBy("label").agg(countDistinct("uid") as "label_uv")
      val df2 = df_label_sku.groupBy("label").agg(count("sku") as "cnt")
      val t0 = df1.join(df2,"label").selectExpr("label","label_uv")

      // when type="sku",collectAsMap  out of memory
      val join_type="mapjoin"
      val t1=join_type match{
        case "mapjoin" =>
          val b = t0.rdd.map(t => (t.getAs("label").toString, t.getAs("label_uv").asInstanceOf[Long])).collectAsMap()
          val bc_b = sc.broadcast(b)
          val a = df_user_label.rdd.map(t => (t.getAs("uid").toString, t.getAs("label").toString, t.getAs("rate").asInstanceOf[Int])).repartition(partitions)
          a.mapPartitions { iter =>
            for {(uid, label, rate) <- iter
                 if (bc_b.value.contains(label))
                 s = bc_b.value.get(label).getOrElse(0L)
            } yield (uid, label, 1.0 * rate / math.log(1 + s))
          }
        case "join" =>
          val tt = df_user_label.join(t0,"label").selectExpr("uid","label","round(rate/log(1+label_uv),4) as rate")
          tt.rdd.map(t=>(t.getAs("uid").toString,t.getAs("label").toString,t.getAs("rate").asInstanceOf[Double]))
      }

//      println("t1.count ="+t1.count())

      val rdd_join =t1.mapPartitions{iter =>
        for{(uid,label,rate) <- iter
            if (bc_t2.value.contains(label))
            skus= bc_t2.value.get(label)
            (sku,score) <- skus.get
        }  yield (uid, sku, math.round(score * rate *10000)/10000.0 )
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

//  def mapjoin[A,B](rdd_a:RDD[(String,A)],rdd_b:RDD[(String,B)],sc:SparkContext ): RDD[(String,A,B)] ={
//    //
//    val t2 = rdd_b.groupByKey().collectAsMap()
//    val bc_t2 = sc.broadcast(t2)
//    val t1=rdd_a
//    t1.mapPartitions{iter =>
//      for{(label,a) <- iter
//          if (bc_t2.value.contains(label))
//          b= bc_t2.value.get(label)
//          b_i <- b.get
//      }yield (label, a, b_i)
//    }
//  }

}
