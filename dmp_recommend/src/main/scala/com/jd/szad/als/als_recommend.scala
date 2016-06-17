package com.jd.szad.als

import com.jd.szad.tools.Writer
import org.apache.spark.mllib.recommendation.{Rating, MatrixFactorizationModel}
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by xieliming on 2016/6/17.
 * 输入sku_id,得到用户报
 * param1 sku_id , param2 require user number ,  param3  user type
 */
object als_recommend {

  def main(args: Array[String]) {
    val sparkConf = new SparkConf()
      .setAppName("ALS_recommend")
      .set("spark.akka.timeout", "500")
      .set("spark.rpc.askTimeout", "500")

    val sc = new SparkContext(sparkConf)

    //val input: String = "app.db/app_szad_m_dmp_als_train/user_type=1/000000_0.lzo"
    val sku_id = args(0).toInt
    val user_num = args(1).toInt
    val user_type = args(2)

    val myModelPath = "app.db/app_szad_m_dmp_als_model/user_type=" + user_type
    val outputPath = "app.db/app_szad_m_dmp_als_sku_res/user_type=" + user_type + "/sku_id=" + sku_id

    //加载模型
    val Model = MatrixFactorizationModel.load(sc, myModelPath)

    //验证sku_id是在训练模型中
    val product = Model.productFeatures.lookup(sku_id)
    if (product.nonEmpty == false) throw new IllegalArgumentException("this sku id not exists the model")
    //recommend
    val res= Model.recommendUsers(sku_id,user_num)

    //sava result
    val res_rdd = sc.parallelize(res,1).map { case Rating(user, product, rate) => user.toString  }
    Writer.write_table(res_rdd,outputPath)

  }


  def toFactor( x : Iterable[(Double,Int)] , k : Int) = {
    val Arr = new Array[Double](k)
    x.foreach{ case(a,b) =>
      Arr(b)=a
    }
    Arr
  }
}
