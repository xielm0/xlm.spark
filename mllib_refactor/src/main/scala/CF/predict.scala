package CF

import com.jd.szad.tools.Writer
import org.apache.spark.sql.SparkSession

/**
 * Created by xieliming on 2016/6/7.
 */
object predict {
  def main(args:Array[String]) {

    val spark = SparkSession.builder()
      .appName("OCPA.CF.predict")
      .config("spark.rpc.askTimeout","500")
      .config("spark.speculation","false")
//      .config("spark.memory.fraction","0.6")
//      .config("spark.memory.storageFraction","0.3")
      .enableHiveSupport()
      .getOrCreate()

    val sc = spark.sparkContext

    val model_type :String = args(0)  // val model_type="train"

    if (model_type =="predict") {

      val user_type=args(1)
      val v_day =args(2).toString
      val v_day_15 = args(3).toString
      val res_path :String = args(4)

      spark.sql(s"set spark.sql.shuffle.partitions = 500")

      //spark2.1 "and length(uid)=32)" has a bug ,so ignore it
      val s1 =
        s"""
           |select uid,sku,rn
           | from app.app_szad_m_dyrec_user_top100_sku
           |where user_type=${user_type} and action_type=1
           |  and sku>0
           |  and dt='${v_day}' and date >'${v_day_15}'
           |  and rn<=10
        """.stripMargin
      val df_apply = spark.sql(s1)

      //      val s2 =
      //        s"""
      //           |select * from(
      //           |select sku1,sku2,cor,
      //           |      row_number() over (partition by uid order by cor desc) rn2
      //           |from app.app_szad_m_dyrec_itemcf_model
      //           |) where rn<=5
      //        """.stripMargin
      val s2 ="select sku1 as sku,sku2,cor from app.app_szad_m_dyrec_itemcf_model_2"
      val df_model = spark.sql(s2)

      val df1 = df_apply.join(df_model,"sku").selectExpr("uid","sku2","cor","rn")
      val res = df1.rdd.map(t=>
        (t.getAs("uid").toString,t.getAs("sku2").toString,t.getAs("cor").toString,t.getAs("rn").toString)
      ).map(t=>
        t._1 +"\t" + t._2 +"\t" + t._3 +"\t" + t._4)

      //save
      Writer.write_table(res,res_path,"lzo")

    }else if (model_type =="predict2") { 
      val user_type=args(1)
      val v_day =args(2).toString
      val v_day_15 = args(3).toString
      val res_path :String = args(4)

      //spark2.1 "and length(uid)=32)" has a bug ,so ignore it
      val s1 =
        s"""
           |select t1.uid,t1.sku,1 rate
           | from app.app_szad_m_dyrec_user_top100_sku t1
           |where user_type=${user_type} and action_type=1
           |  and sku>0
           |  and dt='${v_day}' and date >'${v_day_15}'
           |  and rn<=10
        """.stripMargin
      val user = spark.sql(s1).rdd.map(t=>
        UserPref(t.getAs("uid").toString ,t.getAs("sku").asInstanceOf[Long],t.getAs("rate").asInstanceOf[Int]))

      // get the sim's topk
      val s2 =
        s"""
           |select sku1,sku2,cor from(
           |select sku1,sku2,cor,
           |      row_number() over (partition by uid order by cor desc) rn2
           |from app.app_szad_m_dyrec_itemcf_model
           |) where rn<=5
        """.stripMargin
      val sim = spark.sql(s2).rdd.map(t=>
        ItemSimi(t.getAs("sku1").asInstanceOf[Long] ,t.getAs("sku2").asInstanceOf[Long],t.getAs("cor").asInstanceOf[Double]))

      //预测用户的推荐
      val res = itemCF.recommendItem(sim,user,50)
        .map(t=> t._1 +"\t" + t._2 +"\t" + t._3 +"\t" + t._4)

      //save
      Writer.write_table(res,res_path,"lzo")


    }


  }
}
