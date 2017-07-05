package com.jd.szad.CF


import com.jd.szad.tools.Writer
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

/**
 * Created by xieliming on 2016/6/7.
 * spark-shell --num-executors 80 --executor-memory 10g  --executor-cores 5
 */

object predict {
  def main(args:Array[String]) {

    val spark = SparkSession.builder()
      .appName("DPA.CF.predict")
      .config("spark.rpc.askTimeout","500")
      .config("spark.speculation","false")
//      .config("spark.memory.fraction","0.6")
//      .config("spark.memory.storageFraction","0.3")
      .enableHiveSupport()
      .getOrCreate()

    val model_type :String = args(0)  // val model_type="train"

    val user_type=args(1)
    val v_day =args(2).toString
    val v_day_from = args(3).toString
    val res_path :String = args(4)

    spark.sql(s"set spark.sql.shuffle.partitions = 800")

    //spark2.1 "and length(uid)=32)" has a bug ,so ignore it
    val s1 =
      s"""
         |select uid,sku,rn
         | from app.app_szad_m_dyrec_user_top100_sku
         |where user_type=${user_type} and action_type=1
         |  and sku>0
         |  and dt='${v_day}' and date >'${v_day_from}'
        """.stripMargin
    val df_apply = spark.sql(s1)

    val s2 =
      s"""
         |select sku1 as sku,sku2,cor from(
         |select sku1,sku2,cor,
         |      row_number() over (partition by sku1 order by cor desc) rn
         |from app.app_szad_m_dyrec_itemcf_model
         |)t where rn<=10
        """.stripMargin
    //      val s2 ="select sku1 as sku,sku2,cor from app.app_szad_m_dyrec_itemcf_model_2"
    val df_model = spark.sql(s2)

    // df join
    if (model_type =="predict") {

      val df3 = df_apply.join(df_model,"sku").selectExpr("uid","sku2","cor","rn")
      val df4 = df3.groupBy("uid","sku2").agg(sum("cor") as "cor", min("rn") as "rn" )
      val res = df4.rdd.map(t=>
        (t.getAs("uid").toString +"\t"+  t.getAs("sku2").toString +"\t"+ t.getAs("cor").toString +"\t"+ t.getAs("rn").toString)
      )

      //save
      Writer.write_table(res,res_path,"lzo")

    }else{
      // 解决数据倾斜
      // 找到热门key。通过数据抽样
      val hot =df_apply.sample(false,0.1).groupBy("sku").agg(count("1") as "cnt").select("sku","cnt")
      val hot_n = hot.where(col("cnt") > 100000).rdd.map(t=>t.getAs[Long]("sku")).collect()
//      val hot_n = hot.orderBy("cnt").limit(1000).rdd.map(_.getAs[Long]("sku")).collect()
      val hot_bc = spark.sparkContext.broadcast(hot_n)

      //join
      val df1_1 = df_apply.filter( t=> !hot_bc.value.contains(t.getAs[Long]("sku") ))
      val df2_1 = df_model.filter( t=> !hot_bc.value.contains(t.getAs[Long]("sku") ))
      val df3_1 = df1_1.join(df2_1,"sku").selectExpr("uid","sku2","cor","rn")


      //mapjoin
      val df1_2 = df_apply.filter( t=> hot_bc.value.contains(t.getAs[Long]("sku") )).rdd
        .map(t=>(t.getAs[String]("uid"),t.getAs[Long]("sku"),t.getAs[Double]("rn")))
      val df2_2 = df_model.filter( t=> hot_bc.value.contains(t.getAs[Long]("sku") )).rdd
        .map(t=>(t.getAs[Long]("sku"),(t.getAs[Long]("sku2"),t.getAs[Double]("cor")))).groupByKey().collectAsMap()
      val bc_t2 = spark.sparkContext.broadcast(df2_2)

      val c = df1_2.mapPartitions{iter =>
        for{(uid,sku,rate) <- iter
            if (bc_t2.value.contains(sku))
            vv= bc_t2.value.get(sku)
            (sku2,cor) <- vv.get
        }  yield (uid, sku, rate * cor  )
      }

      import spark.implicits._
      val df3_2 = c.toDF()

      val df4 = df3_1.union(df3_2).groupBy("uid","sku2").agg(sum("cor") as "cor", min("rn") as "rn" )

      val res = df4.rdd.map(t=>
        (t.getAs("uid").toString +"\t"+  t.getAs("sku2").toString +"\t"+ t.getAs("cor").toString +"\t"+ t.getAs("rn").toString)
      )

      //save
      Writer.write_table(res,res_path,"lzo")

    }


  }
}
