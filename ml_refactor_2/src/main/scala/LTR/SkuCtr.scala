package com.jd.szad.LTR

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel

/**
 * Created by xieliming on 2017/6/26.
 */
object SkuCtr {
  def main(args: Array[String]) {
    val spark = SparkSession.builder()
      .appName("CPM.LTR.Ctr")
      .config("spark.rpc.askTimeout", "500")
      .config("spark.speculation","false")
      .enableHiveSupport()
      .getOrCreate()

    val v_day:String = args(0)
    val v_day_from :String=args(1)

    val s1 =
      s"""
         |select a1.adspec_id,
         |       a1.sku_id,
         |       a1.expose_nums,
         |       a1.click_nums,
         |       a1.realcost,
         |       a2.item_third_cate_cd as cate_id
         |  from(select adspec_id,split(sku_id,',')[0] as sku_id,
         |             count( expose_id) expose_nums,
         |             count( click_id) click_nums,
         |             sum( nvl(realcost,0)) realcost
         |        from app.app_szad_m_dyrec_expose_click_1
         |       where media=1
         |         and adspec_id='2000611'
         |         and dt<='${v_day}' and dt >'${v_day_from}'
         |         and length(split(sku_id,',')[0])>2
         |      group by  adspec_id,split(sku_id,',')[0]
         |      )a1
         |join (select * from gdm.gdm_m03_item_sku_da where dt='${v_day}' ) a2
         |  on  a1.sku_Id=a2.item_sku_id
      """.stripMargin

    spark.sql("set hive.optimize.skewjoin=true")
    val df1=spark.sql(s1).persist(StorageLevel.MEMORY_AND_DISK)
    //df1.schema

    // cate ctr
    val df2= df1.groupBy("adspec_id","cate_id")
      .agg(sum("expose_nums") as "expose_nums",sum("click_nums") as "click_nums",sum("realcost") as "cost")
      .selectExpr("adspec_id","cate_id","expose_nums","click_nums","cost")

    val df3= df2.filter(col("expose_nums")>= 1000 and  col("click_nums")>0)
      .selectExpr("adspec_id","cate_id","expose_nums","click_nums","click_nums/expose_nums as p","cost/click_nums as ppc")

    val df4 =df3.selectExpr("adspec_id","cate_id","expose_nums","click_nums","ppc",
      "p- sqrt(p*(1-p)/100)*1.28 as low","p+ sqrt(p*(1-p)/100)*1.28 as up")

    //save
    import spark.implicits._
    df4.createOrReplaceTempView("tmp_cate")
    spark.sql("set mapreduce.output.fileoutputformat.compress=true")
    spark.sql("set hive.exec.compress.output=true")
    spark.sql("set mapred.output.compression.codec=com.hadoop.compression.lzo.LzopCodec")
    spark.sql(
      s"""
         |insert overwrite table app.app_szad_m_dyrec_cate_ctr partition(dt='${v_day}')
         |select adspec_id,cate_id,ppc,expose_nums,click_nums,low,up
         |from tmp_cate
      """.stripMargin)


    val df_Cate=df3.selectExpr("adspec_id","cate_id","100 as unit_expose" , "round(100*click_nums/expose_nums ,2) as unit_click","ppc")

    val rdd_Cate = df_Cate.rdd.map(t=>
      ((t.getAs[String]("adspec_id"),t.getAs[String]("cate_id")),
        (t.getAs[Int]("unit_expose").toLong,t.getAs[Double]("unit_click"),t.getAs[Double]("ppc").toLong))).collectAsMap()


    // rdd mapjoin
    val bc_Cate = spark.sparkContext.broadcast(rdd_Cate)
    val z=1.28
    val rdd_Ctr= df1.rdd.map(t=>
      ( (t.getAs[String]("adspec_id"),t.getAs[String]("cate_id")),
        (t.getAs[String]("sku_id"),t.getAs[Long]("expose_nums"),t.getAs[Long]("click_nums"),t.getAs[Long]("realcost"))))
      .map { case ((adspec_id, cate_id), (sku_id,expose_nums,click_nums,realcost)) =>
        if (bc_Cate.value.contains((adspec_id, cate_id))){
          val (unit_expose,unit_click,bid)=bc_Cate.value.get((adspec_id, cate_id)).getOrElse((0L,0.0,0L))
          (adspec_id, sku_id,cate_id,if (click_nums==0) bid else realcost/click_nums ,expose_nums,click_nums,
            1.0*(click_nums+unit_click)/(expose_nums+unit_expose), expose_nums+unit_expose  )
        }else {
          val (unit_expose,unit_click)=(100,1)
          (adspec_id, sku_id,cate_id, if (click_nums==0) 0 else realcost/click_nums ,expose_nums ,click_nums,
            1.0*(click_nums+unit_click)/(expose_nums+unit_expose), expose_nums+unit_expose )}
      }.map{case(adspec,sku,cate,ppc,expose,clicks,p:Double,n)=>
      (adspec,sku,cate,ppc,expose,clicks, p- z*math.sqrt(p*(1-p)/n) ,p+ z*math.sqrt(p*(1-p)/n)  )
    }

    //save
    val df_Ctr = rdd_Ctr.toDF("adspec_id","sku_id","cate_id","ppc","expose_nums","click_nums","low","up")
    // df_Ctr.stat.approxQuantile("expose_nums",Array(0.25,0.5,0.75,1.0),0)

    df_Ctr.createOrReplaceTempView("tmp")
    spark.sql("set mapreduce.output.fileoutputformat.compress=true")
    spark.sql("set hive.exec.compress.output=true")
    spark.sql("set mapred.output.compression.codec=com.hadoop.compression.lzo.LzopCodec")
    spark.sql(
      s"""
         |insert overwrite table app.app_szad_m_dyrec_sku_ctr partition(dt='${v_day}')
         |select adspec_id,sku_id,cate_id,ppc,expose_nums,click_nums,low,up
         |from tmp
      """.stripMargin)


  }

}
