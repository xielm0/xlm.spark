package com.jd.szad.CF


import com.jd.szad.tools.Writer
import org.apache.spark.sql.SparkSession

/**
 * Created by xieliming on 2016/6/7.
 */
object app {
  def main(args:Array[String]) {
    val spark = SparkSession.builder()
      .appName("DPA.CF")
      .config("spark.rpc.askTimeout","500")
      .config("spark.speculation","false")
      .config("spark.memory.fraction","0.6")
      .config("spark.memory.storageFraction","0.3")
      .enableHiveSupport()
      .getOrCreate()

    val sc = spark.sparkContext

    val model_type :String = args(0)  // val model_type="train"
    val partitions :Int = args(1).toInt
    val model_path :String =args(2)

//    spark.sql(s"set spark.sql.shuffle.partitions = ${partitions}")

    if (model_type =="train") {
      val s0=
        """
          |select user_id,item_id
          |from app.app_szad_m_dyrec_itemcf_train
          |where action_type=1
        """.stripMargin
      val user_action = spark.sql(s0).rdd.repartition(partitions).map(t=>
        UserItem(t.getAs("user_id").toString,t.getAs("item_id").asInstanceOf[Long])
      ).cache()  //.persist(StorageLevel.MEMORY_AND_DISK )

//      val user_action2 = sc.textFile("app.db/app_szad_m_dyrec_itemcf_train/action_type=1").map(t=>
//      t.split("\t") match {case (user_id,item_Id,cate_id) =>UserItem(user_id,item_Id,cate_id) })

      println("user_action count=" + user_action.count())

      //sku
      val s1 =
        s"""select bigint(outerid) sku  from app.app_szad_m_dyrec_sku_list_day group by outerid
       """.stripMargin
      val ad_sku = spark.sql(s1).rdd.map(t=> t.getAs("sku").asInstanceOf[Long]).collect().zipWithIndex.toMap

      //sku-cate map
      // 原本打算map后broadcast,但太大，内存不足
      val s2 =
        """
          |select item_id as sku,max(cate_id) as cate_id
          |from app.app_szad_m_dyrec_itemcf_train
          |where action_type=1
          |group by item_id
        """.stripMargin
      val sku_cate = spark.sql(s2).rdd.map(t=> (t.getAs("sku").asInstanceOf[Long],t.getAs("cate_id").asInstanceOf[Int]))

      //计算相似度
      val similary = itemCF.compute_sim(sc,user_action,ad_sku,sku_cate,10)
        .repartition(100)
        .map(t=>t.itemid1 +"\t" + t.itemid2 +"\t" + t.similar  )

      //保存到hdfs
      Writer.write_table(similary,model_path,"lzo")

    }


  }
}
