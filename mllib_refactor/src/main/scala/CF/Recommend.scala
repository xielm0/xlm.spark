package CF

import org.apache.spark.rdd.RDD

/**
 * Created by xieliming on 2017/6/15.
 */
class Recommend {
  def recommendItem(item_similar : RDD [ItemSimi],
                    user_action : RDD[UserPref],
                    k : Int) ={

    //矩阵相乘
    // 这里join可能会产生数据倾斜。
    val rdd1 = item_similar.map(t=> (t.itemid1,(t.itemid2,t.similar)))
      .join(user_action.map(t=>(t.itemid,(t.userid,t.score))))


    val rdd2 = rdd1.map{case (itemid1,((itemid2,similar),(userid,score))) => ((userid,itemid2) , similar * score)}

    //按(用户,sku)累计求和
    val rdd3 = rdd2.reduceByKey(_+_).map(t=> (t._1,math.round(1000*t._2)/1000.0))

    // rdd3数据很大，对一个用户，若是30*30，则最多有900个sku .此时再join耗费资源非常大。所以先top,再join过滤掉已经浏览/购买的sku.
    //取topk
    val rdd4= rdd3.map{ case((uid,sku),score)=>(uid,(sku,score))}
      .groupByKey()
      .flatMap {case (uid, b) =>
        val topk = b.toArray.sortWith { (a, b) => a._2 > b._2 }.take(k)
        topk.zipWithIndex.map(t => ((uid, t._1._1), (t._1._2, t._2))) //t._2 = rn
      }

    // 过滤掉已经浏览/购买的sku
    rdd4.leftOuterJoin(user_action.map(t=>((t.userid,t.itemid),1)))
      .filter(t=>t._2._2.isEmpty)
      .map(t=> (t._1._1,t._1._2,t._2._1._1,t._2._1._2))

  }


  /*  def df_join(df_user_label :DataFrame , df_label_sku:DataFrame  ,k :Int):DataFrame = {
    //df join
    val df1 = df_user_label.join(df_label_sku, "label").selectExpr("uid", "sku", "rate*score as score")
    val df2 = df1.groupBy("uid", "sku").agg(round(sum("score"), 4) as "score")

    //top 100
    val w = Window.partitionBy("uid").orderBy(desc("score"))
    val df4 = df2.select(col("uid"), col("sku"), col("score"), rowNumber().over(w).alias("rn")).where(s"rn<=${k}")
    df4

  }*/

}
