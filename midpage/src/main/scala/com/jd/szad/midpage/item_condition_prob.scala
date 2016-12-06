package com.jd.szad.midpage

import com.jd.szad.tools._
import org.apache.spark.{SparkConf, SparkContext}


/**
 * Created by xieliming on 2016/9/8.
 */
object item_condition_prob {
  def main(args:Array[String]) {
    val conf = new SparkConf()
      .setAppName("item_condition_prob")
      .set("spark.rpc.askTimeout", "500")
      .set("spark.storage.memoryFraction", "0.1" )

    val sc = new SparkContext(conf)

    //load data
    val last_day = args(0)
    val start_day = DateUtil.addDay(last_day,-30)
    print("start_day =" + start_day)

    val part_num = args(1).toInt

    val hContext = new org.apache.spark.sql.hive.HiveContext(sc)

    val s1 =
      s"""
         |select sku_Id,trace_id,imp_sku_id,click_tag
         |from app.app_szad_m_midpage_ad_imppression_day
         |where dt = '${last_day}'
         |and sku_id <> imp_sku_id
         |and sku_cid3 = imp_sku_cid3
         |and ad_sku_type=1
       """.stripMargin

    val org_data = hContext.sql(s1).rdd.repartition(part_num)


    //codi_prob = ctr = click/impression
    //ctr = (click+1)/(imp+200)  //经统计平均ctr=0.8%左右，平均（sku,imp_sku)每天曝光 180次，点击1.5，默认给一个比平均值更低的值。
    //统计（sku,imp_sku)曝光次数 ,要求大于100，目的：过滤掉垃圾数据，如果低于100，则ctr取默认值。1/200
    val rdd_jump1 = org_data.map(t=>((t(0),t(2)),1)).reduceByKey(_+_).filter(_._2>=100)
    //统计（sku,imp_sku)点击次数
    val rdd_jump2 =  org_data.filter(t=>t(0) != t(2) & t(3)==1).map(t=>((t(0),t(2)),1)).reduceByKey(_+_)

    //calc Conditional probability
    val res = rdd_jump2.join(rdd_jump1)
    .map{case(t,(c1,c2)) => (t._1,t._2, math.round(1000.0 * (c1+1) / (c2+200)) / 1000.0)}
    .filter(_._3 > 0.0)
    .map(t=>t._1 + "\t" + t._2 + "\t" + t._3)
    .repartition(1)

    //save result
    Writer.write_table(res,"app.db/app_szad_m_midpage_item_item_cor2_res","lzo")

  }



}
