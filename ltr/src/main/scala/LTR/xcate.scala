package LTR

import org.apache.spark.sql._

/**
 * Created by xieliming on 2017/9/19.
 */
object xcate {
  def get_apply_input(spark:SparkSession,n:Int):DataFrame ={
    val s2 =
      s"""
         |select Double(target_tag)  label
         |      ,user_id
         |      ,sku_id
         |      ,nvl (sku_ctr                    ,0    )     sku_ctr
         |      ,nvl (sku_ctr_up                 ,0    )     sku_ctr_up
         |      ,nvl (sku_ctr_low                ,0    )     sku_ctr_low
         |      ,nvl (sku_ctr_mid                ,0    )     sku_ctr_mid
         |      ,nvl (sku_ctr_gap                ,0    )     sku_ctr_gap
         |      ,nvl (sku_browse_nums            ,0    )     sku_browse_nums
         |      ,nvl (sku_gen_nums               ,0    )     sku_gen_nums
         |      ,nvl (sku_comment_nums           ,0    )     sku_comment_nums
         |      ,nvl (sku_comment_score          ,0    )     sku_comment_score
         |      ,nvl (sku_comment_good_nums      ,0    )     sku_comment_good_nums
         |      ,nvl (sku_comment_good_rate      ,0    )     sku_comment_good_rate
         |      ,nvl (sku_comment_bad_nums       ,0    )     sku_comment_bad_nums
         |      ,nvl (sku_comment_bad_rate       ,0    )     sku_comment_bad_rate
         |      ,nvl (sku_hotscore               ,0    )     sku_hotscore
         |      ,nvl (sku_cvr                    ,0    )     sku_cvr
         |      ,nvl (sku_jd_prc                 ,0    )     sku_jd_prc
         |      ,nvl (sku_jd_prc_after           ,0    )     sku_jd_prc_after
         |      ,nvl (sku_jd_prc_rate            ,0    )     sku_jd_prc_rate
         |      ,nvl (string(cate_id )           ,'s'  )     cate_id
         |      ,nvl (string(brand_id )          ,'s'  )     brand_id
         |      ,nvl (sex                        ,'s'  )     sex
         |      ,nvl (age                        ,'s'  )     age
         |      ,nvl (carrer                     ,'s'  )     carrer
         |      ,nvl (marriage                   ,'s'  )     marriage
         |      ,nvl (haschild                   ,'s'  )     haschild
         |      ,nvl (province                   ,'s'  )     province
         |      ,nvl (city                       ,'s'  )     city
         |      ,nvl (jd_lev                     ,'s'  )     jd_lev
         |      ,nvl (browse_list_sku_date       ,'s:999'  )     browse_list_sku_date
         |      ,nvl (search_list_sku_date       ,'s:999'  )     search_list_sku_date
         |      ,nvl (fav_list_sku_date          ,'s:999'  )     fav_list_sku_date
         |      ,nvl (car_list_sku_date          ,'s:999'  )     car_list_sku_date
         |      ,nvl (gen_list_sku_date          ,'s:999'  )     gen_list_sku_date
         |      ,nvl (browse_list_cate_date      ,'s:999'  )     browse_list_cate_date
         |      ,nvl (search_list_cate_date      ,'s:999'  )     search_list_cate_date
         |      ,nvl (fav_list_cate_date         ,'s:999'  )     fav_list_cate_date
         |      ,nvl (car_list_cate_date         ,'s:999'  )     car_list_cate_date
         |      ,nvl (gen_list_cate_date         ,'s:999'  )     gen_list_cate_date
         |  from (select * from app.app_szad_m_dyrec_rank_apply_new where user_type=1 and n=${n} ) a
         | join (select item_gen_third_cate_id from dim.dim_item_gen_third_cate where item_gen_first_cate_id in('1315','11729','1319','6728'))b
         |   on a.cate_id=b.item_gen_third_cate_id
      """.stripMargin

    spark.sql("set spark.sql.shuffle.partitions =2000")
    spark.sql("set hive.auto.convert.join=true")
    val apply_data = spark.sql(s2)
    apply_data

  }


  def main(args: Array[String]) {
    val spark = SparkSession.builder()
      .appName("CPM.LTR.xcate")
      .config("spark.rpc.askTimeout", "500")
      .config("spark.speculation", "false")
      .config("spark.memory.fraction", "0.8")
      .config("spark.memory.storageFraction", "0.5")
      .enableHiveSupport()
      .getOrCreate()

    val model_stage: String = args(0)
    val n:Int =args(1).toInt
    val apply_data = get_apply_input(spark,n)
    val predictions = lr.predict_res(apply_data,n)

    val model_id:String = args(2)

    // top10
    val k=10

    predictions.createOrReplaceTempView("tmp_table")
    spark.sql("set spark.sql.shuffle.partitions =200")
    spark.sql("use app")
    spark.sql("set mapreduce.output.fileoutputformat.compress=true")
    spark.sql("set hive.exec.compress.output=true")
    spark.sql("set mapred.output.compression.codec=com.hadoop.compression.lzo.LzopCodec")
    val ss =
      s"""
         |insert overwrite table app.app_szad_m_dyrec_rank_predict_res partition(model_id=${model_id},user_type=1)
         |select user_id,sku_Id,prediction,rn
         |from(select user_id,sku_Id,prediction,
         |            row_number() over (partition by user_id order by prediction desc )  rn
         |      from tmp_table
         |    )t
         |where rn <= ${k}
      """.stripMargin
    spark.sql(ss)


  }

}
