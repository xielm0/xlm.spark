package com.jd.szad.als

/**
 * Created by xieliming on 2016/3/31.
 */

import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.{SparkConf, SparkContext}
import com.jd.szad.tools.Writer

object als {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf()
      .set("spark.app.name","ALS_train")
      .set("spark.akka.timeout", "500")
      .set("spark.rpc.askTimeout", "500")

    val sc = new SparkContext(sparkConf)

    //val input: String = "app.db/app_szad_m_dmp_als_train/user_type=1"
    val input: String = args(0)
    val rank = args(1).toInt //秩 1000
    val numIterations = args(2).toInt //20
    val partition_num = args(3).toInt
    val output:String = args(4)

    val org_data = sc.textFile(input).repartition(partition_num).map(_.split("\t"))   //.sample(false, 0.1)

    //数据处理，转变成int
    val users = org_data.map(t=>t(0)).distinct().zipWithIndex().map(t=>(t._1,t._2.toInt)).collectAsMap()
    val skus = org_data.map(t=>t(1)).distinct().zipWithIndex().map(t=>(t._1,t._2.toInt)).collectAsMap()

    val ratings = org_data.map{case Array(user,sku,rate) =>Rating(users(user),skus(sku),rate.toDouble)}

    //train
    //blocks不能太少，否则train不出来结果。 block大于100，则在recommend时，笛卡尔不能接受
    val model: MatrixFactorizationModel = ALS.train(ratings, rank, numIterations, lambda=0.01)

    //评估
    val usersProducts = ratings.map { case Rating(user, product, rate) => (user, product) }
    val predictions =model.predict(usersProducts)
      .map { case Rating(user, product, rate) => ((user, product), rate)  }
    val ratesAndPreds = ratings
      .map { case Rating(user, product, rate) => ((user, product), rate) }
      .join(predictions)
    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean()
    println("Mean Squared Error = " + MSE)
    //

    //recommend
    //全量计算根本计算不出来。
    val res_rdd = model.recommendProductsForUsers(20)
      .flatMap{ case (user, array_rating) =>
                  array_rating.map { case Rating(a, b, c) => a +"\t" + b + "\t" + c }
                }

    //    思路2：
    //    找出用户列表，循环，每次调用model.recommendProducts(user_id,num)得到推荐的结果。
    //    推荐的核心计算是利用 userFeatures x productFeatures的转置 计算得分rate,.
    //    在这里注意，recommendProducts已经调用了一个rdd: userFeatures，所以user不能是RDD，因为RDD不能嵌套使用。
    //    rdd1.map(x => rdd2.values.count() * x) is invalid because the values transformation and count action cannot be performed inside of the rdd1.map transformation.
    /*    val users = ratings.map { case Rating(user, product, rate) => user }.distinct(10).take(10000) //.collect()
        val res = users.flatMap(
            user => model.recommendProducts(user, 5) //return  Array[Rating]
          )
        val res_rdd = sc.makeRDD(res,100).map { case Rating(user, product, rate) => (user, product, rate) }
    */


    //保存推荐结果
    Writer.write_table(res_rdd,output,"lzo")

  }
}
