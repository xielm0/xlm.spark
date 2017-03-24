package com.jd.szad.userlabel

import org.apache.spark.{SparkContext, SparkConf}

import scala.util.Random
import com.jd.szad.tools.Writer


/**
 * Created by xieliming on 2017/3/14.
 */
object catehot {
  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setAppName("cate")
      .set("spark.akka.timeout", "1000")
      .set("spark.rpc.askTimeout", "500")

    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.hive.HiveContext(sc)

    //params
    val input_path = args(0).toString //"app.db/app_szad_m_dyrec_userlabel_apply/user_type=1/type=short_cate"
    val hot_type=args(1)
    val k= args(2).toInt  //20
    val output_path = args(3).toString
    val partitions =500
    val n=3

    //赋值一个随机数
//    val r = new Random()
//    println(s"app.db/app_szad_m_dyrec_userlabel_apply/${input_path}" )
    val org_a = sc.textFile( s"app.db/app_szad_m_dyrec_catehot_apply/${input_path}" ).repartition(partitions)
    val apply = org_a.map(_.split("\t")).mapPartitions{ iter =>
      val r= new Random()
      iter.flatMap{ case Array(uid,cate,rate,rn) =>
        for (i <- 1 to n) yield((cate,r.nextInt(k)+1),(uid,rn.toInt))
      }
    }

    val sql2 =
    s"""
      |select item_third_cate_cd,sku_id,rn
      | from app.app_szad_m_dyrec_catehot_model
      |where  hot_type='${hot_type}' and rn <= ${k}
    """.stripMargin
    val df_cate = sqlContext.sql(sql2)

    val rdd_cate = df_cate.rdd.map(t=>((t.getAs("item_third_cate_cd").toString,t.getAs("rn").asInstanceOf[Int]),t.getAs("sku_id").toString ) )

    val res = apply.join(rdd_cate).map { case (cate,((uid,rn),sku)) => (uid,sku,rn) }

//    val bc_t2 = sc.broadcast(rdd_cate.collectAsMap())
//    val res1 = apply.mapPartitions{iter =>
//      for{((key1:String,key2:Int),uid,rn) <- iter
//          if (bc_t2.value.contains((key1,key2)))
//          sku= bc_t2.value.get((key1,key2)).get
//      }  yield (uid, sku, rn )
//    }

     //save
     val res2 = res.map(t=>t._1 + "\t" + t._2.toString + "\t" + "8" + "\t"+ t._3.toString  )
     Writer.write_table( res2 ,s"app.db/app_szad_m_dyrec_model_predict_res/${output_path}","lzo")


  }

}
