package com.jd.szad.itemcf

import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by xieliming on 2016/5/30.
 */
object predict_itemcf {
  def main(args:Array[String]) {
    val conf = new SparkConf().setAppName("predict_itemCF")
      .set("spark.akka.timeout", "1000")
      .set("spark.rpc.askTimeout", "500")
      .set("spark.storage.memoryFraction","0.1")  //not use cache(),so not need storage memory
    //.set("spark.shuffle.memoryFraction","0.3")

    val input_path :String = args(0)
    val part_num :Int = args(1).toInt
    val output_path:String = args(2)

    val sc = new SparkContext(conf)


  }
}
