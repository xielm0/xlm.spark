package com.jd.szad.als

/**
 * Created by xieliming on 2016/3/31.
 */

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{Path, FileSystem}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

/*
 nohup spark-submit --master yarn-client \
 --class com.jd.szad.recommend.als  \
 --conf spark.dynamicAllocation.enabled=true  --conf spark.shuffle.service.enabled=true --conf spark.dynamicAllocation.maxExecutors=100 \
 --executor-memory 12g --executor-cores 4 \
 --conf spark.shuffle.memoryFraction=0.3 \
 /home/jd_ad/data_dir/xieliming/spark_szad_label_xlm.jar \
 app.db/app_szad_m_dmp_als_train/user_type=1 20 5 800 app.db/app_szad_m_dmp_als_res/user_type=1 app.db/app_szad_m_dmp_als_model/user_type=1 &

 --num-executors 100 --executor-memory 12 --executor-cores 4 \
  */
object als {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf()
      .setAppName("ALS_recommend")
      .set("spark.akka.timeout", "500")
      .set("spark.rpc.askTimeout", "500")
    //.set("spark.akka.frameSize", "128")

    val sc = new SparkContext(sparkConf)

    //val input: String = "app.db/app_szad_m_dmp_als_train/user_type=1/000000_0.lzo"
    val input: String = args(0)
    val rank = args(1).toInt //秩 20
    val numIterations = args(2).toInt //20
    val partition_num = args(3).toInt
    val output:String = args(4)
    val myModelPath:String = args(5)

    val data = sc.textFile(input).repartition(partition_num)   //.sample(false, 0.1)

    val ratings = data.map(_.split("\t") match  {case Array(user,item,rate) =>Rating(user.toInt,item.toInt,rate.toDouble)} )
      .persist(StorageLevel.DISK_ONLY)

    //lambda=0.01 //lambad是用来防止过拟合的。
    val model: MatrixFactorizationModel = ALS.train(ratings, rank, numIterations, 0.01, 100 )

    //隐式反馈
    //    val alpha=0.01
    //    val model = ALS.trainImplicit(ratings, rank, numIterations, lambda, alpha)

    //训练时用来评估的
    //    val usersProducts = ratings.map { case Rating(user, product, rate) =>
    //      (user, product)
    //    }
    //    val predictions =
    //      model.predict(usersProducts).map { case Rating(user, product, rate) =>
    //        ((user, product), rate)
    //      }
    //    val ratesAndPreds = ratings.map { case Rating(user, product, rate) =>
    //      ((user, product), rate)
    //    }.join(predictions)
    //    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
    //      val err = (r1 - r2)
    //      err * err
    //    }.mean()
    //    println("Mean Squared Error = " + MSE)
    //MSE: Double = 0.7326887970584651

    //     Save and load model
    //     save userFeatures and productFeatures
    val conf = new Configuration()
    val fs = FileSystem.get(conf)

    fs.delete(new Path( myModelPath ),true)
    model.save(sc, myModelPath)

    // val sameModel = MatrixFactorizationModel.load(sc, myModelPath)

    //    推荐
    //    思路1：
    //    批量推荐，直接使用 recommendProductsForUsers，返回(user, ratings) ,1.5.2后可以使用
    val res_rdd = model.recommendProductsForUsers(6).flatMap { case (user, array_rating) =>
      array_rating.map { case Rating(a, b, c) => a +"\t" + b + "\t" + c }
    }


    //    思路21：
    //    找出用户列表，循环，每次调用model.recommendProducts(user_id,num)得到推荐的结果。
    //    推荐的核心计算是利用 userFeatures x productFeatures的转置 计算得分rate,.
    //    在这里注意，recommendProducts已经调用了一个rdd: userFeatures，所以user不能是RDD，因为RDD不能嵌套使用。
    //    rdd1.map(x => rdd2.values.count() * x) is invalid because the values transformation and count action cannot be performed inside of the rdd1.map transformation.
    /*    val users = ratings.map { case Rating(user, product, rate) => user }.distinct(10).take(10000) //.collect()
        val res = users.flatMap(
            user => model.recommendProducts(user, 10) //return  Array[Rating]
          )
        val res_rdd = sc.makeRDD(res).map { case Rating(user, product, rate) => (user, product, rate) }
    */


    //保存推荐结果

    fs.delete(new Path( output ),true)
    res_rdd.saveAsTextFile(output)


  }
}
