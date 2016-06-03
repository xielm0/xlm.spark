package com.jd.szad.itemcf

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.rdd.RDD

/**
 * Created by xieliming on 2016/5/30.
 */
object predict_itemcf {
  def recommend(item_similar : RDD [ItemSimi],
                user_action : RDD[UserPref],
                k : Int) ={
    //矩阵相乘
    val rdd1 = item_similar.map(t=> (t.itemid1,(t.itemid2,t.similar)))
      .join(user_action.map(t=>(t.itemid,(t.userid,t.score))))

    val rdd2 = rdd1.map{case (itemid1,((itemid2,similar),(userid,score))) => ((userid,itemid2) , similar * score)}

    //按用户累计求和
    val rdd3 = rdd2.reduceByKey(_+_)

    //过滤掉已有的物品
    val rdd4 = rdd3.leftOuterJoin(user_action.map(t=>((t.userid,t.itemid),1)))
      .filter(t=>t._2._2.isEmpty).map(t=> (t._1._1,(t._1._2,t._2._1)))
    //取topk
    val rdd5= rdd4.groupByKey()

    rdd5.flatMap{
      case(a,b) =>
        val topk=b.toArray.sortWith{ (a,b) => a._2 > b._2 }.take(k)
        topk.zipWithIndex.map(t=> (a,t._1._1,t._1._2,t._2))
    }
  }

  def main(args:Array[String]) {
    val conf = new SparkConf().setAppName("predict_itemCF")
      .set("spark.akka.timeout", "1000")
      .set("spark.rpc.askTimeout", "500")
      .set("spark.storage.memoryFraction","0.1")
    //.set("spark.shuffle.memoryFraction","0.3")

    val sc = new SparkContext(conf)

    val input_path :String = args(0)
    val part_num :Int = args(1).toInt
    val output_path:String = args(2)

    val sim = sc.textFile(input_path,part_num).map(_.split("\t")).map(t=>ItemSimi(t(1).toLong ,t(2).toLong,t(3).toDouble))
    val user = sc.textFile(input_path,part_num).map(_.split("\t")).map(t=>UserPref(t(1) ,t(2).toLong,t(3).toInt))

    val res = recommend(sim,user,5)

    //保存到数据库
    res.map(t=> t._1 +"\t" + t._2 +"\t" + t._3 +"\t" + t._4)
    .saveAsTextFile(output_path)

  }

}
