package org.apache.spark.mllib.recommend

/**
 * Created by xieliming on 2016/3/31.
 */

import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.{SparkConf, SparkContext}

object Als_recommend {
  def main(args:Array[String]) {
    val conf = new SparkConf().setAppName("Als_recommend")
      .set("spark.akka.timeout", "500")
      .set("spark.driver.maxResultSize", "16g")
    val sc = new SparkContext(conf)

    val input: String = args(0) //input data path，livsvm format
    val output:String = args(4) // save prediction and label
    val numIterations = args(2).toInt
    val rank =  args(2).toInt  //秩

    //val data = MLUtils.loadLibSVMFile(sc,input)
    val data = sc.textFile("data/mllib/als/test.data")
    val ratings = data.map(_.split(',') match { case Array(user, item, rate) =>
      Rating(user.toInt, item.toInt, rate.toDouble)
    })

    val model = ALS.train(ratings, rank, numIterations, 0.01)

    val usersProducts = ratings.map { case Rating(user, product, rate) =>
      (user, product)
    }
    val predictions =
      model.predict(usersProducts).map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }
    val ratesAndPreds = ratings.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(predictions)
    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean()
    println("Mean Squared Error = " + MSE)

    // Save and load model
    model.save(sc, "myModelPath")
    val sameModel = MatrixFactorizationModel.load(sc, "myModelPath")

  }
}
