package com.jd.szad.itemcf

import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by xieliming on 2016/6/7.
 */
object app {
  def main(args:Array[String]) {
    val conf = new SparkConf().setAppName("itemCF")
      .set("spark.akka.timeout", "1000")
      .set("spark.rpc.askTimeout", "500")
      .set("spark.storage.memoryFraction","0.1")  //not use cache(),so not need storage memory
    //.set("spark.shuffle.memoryFraction","0.3")

    val sc = new SparkContext(conf)

    val model_type :String = args(0)  // val model_type="train"
    val input_path :String = args(1)
    val part_num :Int = args(2).toInt
    val model_path :String =args(3)
    val output_path:String = args(4)

    if (model_type =="train") {
      val data=sc.textFile(input_path).repartition(part_num)

      val user_item = data.map( _.split("\t") match{ case Array(user,item,rate) =>UserItem(user,item.toLong)})

      //计算相似度
      val similary = itemCF.compute_sim(user_item,part_num,10)
        .map(t=>t.itemid1 +"\t" + t.itemid2 +"\t" + t.similar)

      //保存到hdfs
      Writer.write_table(similary,model_path)

    } else if (model_type =="predict") {
      val user = sc.textFile(input_path,part_num).map(_.split("\t")).map(t=>UserPref(t(0) ,t(1).toLong,t(2).toInt))

      val sim = sc.textFile(model_path).map(_.split("\t")).map(t=>ItemSimi(t(0).toLong ,t(1).toLong,t(2).toDouble))

      //预测用户的推荐
      val res = predict_itemcf.recommend(sim,user,5)
        .map(t=> t._1 +"\t" + t._2 +"\t" + t._3 +"\t" + t._4)

      //保存到hdfs
      Writer.write_table(res,output_path)

    }


  }
}
