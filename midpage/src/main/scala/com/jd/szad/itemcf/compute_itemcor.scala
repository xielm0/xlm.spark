package com.jd.szad.itemcf

import com.jd.szad.tools.Writer
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by xieliming on 2016/6/7.
 */
object compute_itemcor {
  def main(args:Array[String]) {
    val conf = new SparkConf()
      .setAppName("compute_item_cor")
      .set("spark.akka.timeout", "1000")
      .set("spark.rpc.askTimeout", "500")

    val sc = new SparkContext(conf)

    val model_type :String = args(0)  // val model_type="train"
    val input_path :String = args(1)
    val part_num :Int = args(2).toInt
    val model_path :String =args(3)
    val last_day = args(4) // val last_day =2016-09-18

    if (model_type =="train") {
      val data=sc.textFile(input_path).repartition(part_num)

      val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)
      val s1 =
        s"""select adid from app.app_szad_m_midpage_valid_ad where dt='${last_day}'
       """.stripMargin
      val ad_sku = hiveContext.sql(s1).map(t=> t(0).asInstanceOf[Long]).collect().zipWithIndex.toMap

      val user_action = data.map( _.split("\t") match{ case Array(user,item,rate) =>UserItem(user,item.toLong)})

      //计算相似度
      val similary = itemCF.compute_sim(sc,user_action,ad_sku)
        .repartition(50).map(t=>t.itemid1 +"\t" + t.itemid2 +"\t" + t.similar  )

      //保存到hdfs
      Writer.write_table(similary,model_path,"lzo")

    } else if (model_type =="predict") {

      val user = sc.textFile(input_path).repartition(part_num).map{t=>
        val x=t.split("\t")
        if (x(1) != "\\N") UserPref(x(0) ,x(1).toLong,x(4).toInt)
        else UserPref(x(0) ,0,x(4).toInt)
      }

      val sim = sc.textFile(model_path).map(_.split("\t")).map(t=>ItemSimi(t(0).toLong ,t(1).toLong,t(2).toDouble))

      //预测用户的推荐
      val res = predict_itemcf.recommend(sim,user,20)
        .map(t=> t._1 +"\t" + t._2 +"\t" + t._3 +"\t" + t._4)

      //保存到hdfs
      //Writer.write_table(res,args(4))

    }


  }
}
