package com.jd.szad.LTR

import org.apache.spark.sql.SparkSession

/**
 * Created by xieliming on 2017/5/16.
 * spark-shell --num-executors 5 --executor-memory 8g --executor-cores 4
 */
object pino {

  def main(args: Array[String]) {
    val spark = SparkSession.builder()
      .appName("pino_test")
      .config("spark.rpc.askTimeout", "500")
      .config("spark.speculation", "false")
      .config("spark.memory.fraction", "0.6")
      .config("spark.memory.storageFraction", "0.3")
      .enableHiveSupport()
      .getOrCreate()

    // 将测试数据转化成pino的格式
    val s1=
      """
        |select case when click_tag=1 then 1 else -1 end as tag,
        |sku_id,sku_cid3,imp_sku_id,imp_sku_cid3,if_cid3,nvl(commentnum,0)commentnum,good_rate,
        |nvl(algo1_score,0) as algo1,nvl(algo2_score,0) as algo2,nvl(algo3_score,0) as algo3
        |from app.app_szad_m_midpage_mixrank_train
        |where dt='2016-11-20'
      """.stripMargin

    val rdd1=spark.sql(s1).rdd.map{t=>
      t.getAs("tag").toString + " 1|a1 "+ t.getAs("sku_id")+"|a2 "+ t.getAs("sku_cid3")+"|a3 "+ t.getAs("imp_sku_id")+"|a4 "+
        t.getAs("imp_sku_cid3")+"|a5 "+ t.getAs("if_cid3")+"|b1 "+
        t.getAs("commentnum")+"|b2 "+t.getAs("good_rate")+"|b3 "+t.getAs("algo1")+"|b4 "+t.getAs("algo2")+"|b5 "+t.getAs("algo3")
    }



    //save
    rdd1.repartition(20).saveAsTextFile("/user/jd_ad/ads_sz/app.db/app_szad_m_dyrec_pino/20170501")



  }

}
