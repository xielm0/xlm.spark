package com.jd.szad.CF

import com.jd.szad.tools.Writer
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by xieliming on 2016/6/7.
 */
object app {
  def main(args:Array[String]) {
    val conf = new SparkConf()
      .setAppName("itemCF")
      .set("spark.akka.timeout", "1000")
      .set("spark.rpc.askTimeout", "500")

    val sc = new SparkContext(conf)

    val model_type :String = args(0)  // val model_type="train"
    val input_path :String = args(1)
    val part_num :Int = args(2).toInt
    val model_path :String =args(3)
    val res_path :String = args(4)

    val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)

    if (model_type =="train") {

      val data=sc.textFile(input_path).repartition(part_num)

      //sku
      val s1 =
        s"""select bigint(outerid) from app.app_szad_m_dyrec_sku_list_day
       """.stripMargin
      val ad_sku = hiveContext.sql(s1).map(t=> t(0).asInstanceOf[Long]).collect().zipWithIndex.toMap

      val user_action = data.map( _.split("\t") match{ case Array(user,item,rate) =>UserItem(user,item.toLong)})

      //计算相似度
      val similary = itemCF.compute_sim(sc,user_action,ad_sku)
        .repartition(100).map(t=>t.itemid1 +"\t" + t.itemid2 +"\t" + t.similar  )

      //保存到hdfs
      Writer.write_table(similary,model_path,"lzo")

    } else if (model_type =="predict") {

      val user = sc.textFile(input_path).repartition(part_num)
        .map(_.split("\t"))
        .map{x=>
        if (x(1) != "\\N") UserPref(x(0) ,x(1).toLong, x(2).toInt)
        else UserPref(x(0) ,0L,x(2).toInt)
      }

      val sim = sc.textFile(model_path).map(_.split("\t")).map(t=>ItemSimi(t(0).toLong ,t(1).toLong,t(2).toDouble))

      //预测用户的推荐
      val res = itemCF.recommendItem(sim,user,20)
        .map(t=> t._1 +"\t" + t._2 +"\t" + t._3 +"\t" + t._4)

      //保存到hdfs
      Writer.write_table(res,res_path,"lzo")

    }


  }
}
