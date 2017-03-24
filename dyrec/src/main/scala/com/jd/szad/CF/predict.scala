package com.jd.szad.CF

import com.jd.szad.tools.Writer
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by xieliming on 2016/6/7.
 */
object predict {
  def main(args:Array[String]) {
    val conf = new SparkConf()
      .setAppName("OCPA.cf.predict")
      .set("spark.akka.timeout", "1000")
      .set("spark.rpc.askTimeout", "500")

    val sc = new SparkContext(conf)

    val model_type :String = args(0)  // val model_type="train"

    val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)

    if (model_type =="predict") {
      // 给多个sku进行推荐，不能按相关性做排序，只能作为备选集
      val user_type=args(1)
      val v_day =args(2).toString
      val v_day_15 = args(3).toString
      val res_path :String = args(4)

      val s1 =
        s"""
           |select t1.uid,t1.sku,t1.rn
           | from app.app_szad_m_dyrec_user_top100_sku t1
           |where user_type=${user_type} and action_type=1
           |  and length(uid)='32' and sku>0
           |  and dt='${v_day}' and date >'${v_day_15}'
           |  and rn<=10
        """.stripMargin
      val df_apply = hiveContext.sql(s1)

      //      val s2 =
      //        s"""
      //           |select * from(
      //           |select sku1,sku2,cor,
      //           |      row_number() over (partition by uid order by cor desc) rn2
      //           |from app.app_szad_m_dyrec_itemcf_model
      //           |) where rn<=5
      //        """.stripMargin
      val s2 ="select sku1 as sku,sku2,cor from app.app_szad_m_dyrec_itemcf_model_2"
      val df_model = hiveContext.sql(s2)

      val df1 = df_apply.join(df_model,"sku").selectExpr("uid","sku2","cor","rn")
      val res = df1.rdd.map(t=>(t.getAs("uid").toString,t.getAs("sku2"),t.getAs("cor"),t.getAs("rn"))).map(t=>
        t._1 +"\t" + t._2 +"\t" + t._3 +"\t" + t._4)

      //保存到hdfs
      Writer.write_table(res,res_path,"lzo")

    }else if (model_type =="predict2") { 
      val user_type=args(1)
      val v_day =args(2).toString
      val v_day_15 = args(3).toString
      val res_path :String = args(4)

      val s1 =
        s"""
           |select t1.uid,t1.sku,1 rate
           | from app.app_szad_m_dyrec_user_top100_sku t1
           |where user_type=${user_type} and action_type=1
           |  and length(uid)='32' and sku>0
           |  and dt='${v_day}' and date >'${v_day_15}'
           |  and rn<=10
        """.stripMargin
      val user = hiveContext.sql(s1).rdd.map(t=>
        UserPref(t.getAs("uid").toString ,t.getAs("sku").asInstanceOf[Long],t.getAs("rate").asInstanceOf[Int]))

      // 取sim的top k
      val s2 =
        s"""
           |select sku1,sku2,cor from(
           |select sku1,sku2,cor,
           |      row_number() over (partition by uid order by cor desc) rn2
           |from app.app_szad_m_dyrec_itemcf_model
           |) where rn<=5
        """.stripMargin
      val sim = hiveContext.sql(s2).rdd.map(t=>
        ItemSimi(t.getAs("sku1").asInstanceOf[Long] ,t.getAs("sku2").asInstanceOf[Long],t.getAs("cor").asInstanceOf[Double]))

      //预测用户的推荐
      val res = itemCF.recommendItem(sim,user,50)
        .map(t=> t._1 +"\t" + t._2 +"\t" + t._3 +"\t" + t._4)

      //保存到hdfs
      Writer.write_table(res,res_path,"lzo")


    }


  }
}
