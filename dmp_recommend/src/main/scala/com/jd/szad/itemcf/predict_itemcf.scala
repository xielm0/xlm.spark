package com.jd.szad.itemcf

import org.apache.spark.rdd.RDD

/**
 * Created by xieliming on 2016/5/30.
 */
object predict_itemcf {
  def recommend(item_similar : RDD [ItemSimi],
                user_action : RDD[UserPref],
                k : Int) ={

    //矩阵相乘
    val rdd1 = user_action.map(t=>(t.itemid,(t.userid,t.score)))
      .join(item_similar.map(t=> (t.itemid1,(t.itemid2,t.similar))))

    val rdd2 = rdd1.map{case (itemid1,((userid,score),(itemid2,similar))) => ((userid,itemid2) , similar * score)}

    //按用户累计求和
    val rdd3 = rdd2.reduceByKey(_+_).map(t=> (t._1,math.round(1000*t._2)/1000.0))

    //过滤掉已有的物品
    val rdd4 = rdd3.leftOuterJoin(user_action.map(t=>((t.userid,t.itemid),1)))
      .filter(t=>t._2._2.isEmpty).map(t=> (t._1._1,(t._1._2,t._2._1)))

    //取topk
    val rdd5= rdd4.groupByKey()

    rdd5.flatMap{
      case(a,b) =>
        val topk=b.toArray.sortWith{ (a,b) => a._2 > b._2 }.take(k)
        topk.zipWithIndex.map(t=> (a,t._1._1,t._1._2,t._2))   //t._2 = rn
    }

  }


}
