package com.jd.szad.als

import breeze.numerics.ceil
import org.apache.spark.mllib.recommendation.{MatrixFactorizationModel, Rating}
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by xieliming on 2016/6/17.
 * 输入sku_id,得到用户报
 * param1 sku_id , param2 require user number ,  param3  user type
 */
object als_recommend {

  def main(args: Array[String]) {
    val sparkConf = new SparkConf()
      .set("spark.app.name","ALS_recommend")
      .set("spark.akka.timeout", "500")
      .set("spark.rpc.askTimeout", "500")
      .set("spark.storage.memoryFraction","0.05")
//      .set("spark.akka.frameSize", "128")

    val sc = new SparkContext(sparkConf)

    //val input: String = "app.db/app_szad_m_dmp_als_train/user_type=1/000000_0.lzo"
    val sku_id = args(0)
    val user_num = args(1).toInt
    val user_type = args(2).toInt

    val myModelPath = "app.db/app_szad_m_dmp_als_model/user_type=" + user_type
    //val outputPath = "app.db/app_szad_m_dmp_als_sku_res/user_type=" + user_type + "/" + sku_id + ".txt"

    //sku_id 转化为 int型
    val sku_path = "app.db/app_szad_m_dmp_sku2int/dt=*/*.lzo"
    val rdd_sku = sc.textFile(sku_path,100).map(_.split("\t")).map(t=> (t(0),t(1).toInt))
    val sku_int:Int = rdd_sku.lookup(sku_id).head
    //加载模型
    val Model = MatrixFactorizationModel.load(sc, myModelPath)

    //验证sku_id是在训练模型中
    val product = Model.productFeatures.lookup(sku_int)
    if (product.nonEmpty == false) throw new IllegalArgumentException("this sku id not exists the model, sku id =" + sku_int)
    //recommend
    val res= Model.recommendUsers(sku_int,user_num)

    //user转换成string
    val user_path = "app.db/app_szad_m_dmp_user2int/user_type=" + user_type + "/dt=*/*.lzo"
    val rdd_user = sc.textFile(user_path,40).map(_.split("\t")).map(t=> (t(1).toInt ,t(0)))

    //sav3eresult
    val parts =ceil(user_num/1000000)
    val res_rdd = sc.makeRDD(res,parts)
      .map { case Rating(user, product, rate) => (user,1)  }

    val res_rdd2=rdd_user.join(res_rdd)
      .map(t=>t._2._1)
      .repartition(parts)

    println("recommend user2 num is " + res_rdd2.count())

//    Writer.write_table(res_rdd.repartition(1),outputPath)
    val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)
    import hiveContext.implicits._
    res_rdd2.toDF("user_id").registerTempTable("tmp_table")
    val sql = "insert overwrite table app.app_szad_m_dmp_als_sku_res partition(user_type="+user_type +",sku_id=" + sku_id +") "+
    " select user_id from tmp_table"
    hiveContext.sql(sql)

  }


  def toFactor( x : Iterable[(Double,Int)] , k : Int) = {
    val Arr = new Array[Double](k)
    x.foreach{ case(a,b) =>
      Arr(b)=a
    }
    Arr
  }
}
