
/**
 * Created by xieliming on 2016/4/21.
 */
package com.jd.szad.itemcf

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel


object itemCF {

  /*分别计算每个user的(item,item)相似性的分子
  输入x=(user,(item1,item2,...))
  * */
  //1个user如果有100个商品，则产生100*100对tuple,相当于笛卡尔
  def compute_numerator(item_seq :Iterable[Long],ad_sku :Map[Long,Int] )={
    val item_cnt = item_seq.map(t=> 1).reduce(_+_)
    for (item1 <- item_seq;
         item2<- item_seq;
         if item1 != item2 & ad_sku.contains(item2))
      yield { ((item1,item2),1.0/math.log(1 + item_cnt))}
  }


  def compute_sim( sc:SparkContext ,user_action :RDD[UserItem]  ,ad_sku :Map[Long,Int] ): RDD[ItemSimi] ={

    val ad_sku_b = sc.broadcast(ad_sku)

    //为了节省内存，将item由字符串转换为int型。
//    val item_map = user_action .map (t=> t.itemid).distinct(100).zipWithIndex().map(t=> (t._1,t._2.toInt)).cache()
//    print("item count is " +item_map.count() )

    val user_item = user_action.map(t=>(t.userid,t.itemid)).persist(StorageLevel.MEMORY_AND_DISK )
    print( " user_item cnt  = "+user_item.count())

    val item_freq = user_item.map{t=>(t._2,1)}.reduceByKey(_+_)
    val item_freq_b=sc.broadcast(item_freq)

    val item_pair = user_item.groupByKey().flatMap{x=> compute_numerator(x._2,ad_sku_b.value)}.persist(StorageLevel.MEMORY_AND_DISK_SER )

    //相似性计算的分子
    val sim_numerator = item_pair.reduceByKey(_+_)
    print ( " sim_numerator cnt  = " + sim_numerator.first().toString())

    //计算交集
    val item_join = item_pair.map(t=>(t._1,1)).reduceByKey(_+_)

    //相似性计算的分母，即并集.返回 ((item_1,item2),score)
    //并集=item1集合 + item2集合 - 交集
    val sim_denominator = item_join.map{ case ((item1,item2),num) => (item1,(item2,num)) }
      .join(item_freq_b.value).map{case  (item1,((item2,num),fre1)) => (item2,(item1,fre1,num))}
      .join(item_freq_b.value).map{case  (item2,((item1,fre1,num),fre2)) =>
      ((item1,item2) , fre1 + fre2 - num)
    }
    //相似度

    //过滤掉极小值
    val sim= sim_numerator.join(sim_denominator).map{
      case((item1,item2),(num,denom)) =>
        val cor = if (num/denom>0.0001)  num/denom else 0.0001
        (item1,item2 ,math.round(10000 * cor)/1000.0 )
    }.filter(_._3>0.001)

    //top k
//    val sim_topk = sim.map{case (item1,item2,score) => (item1,(item2,score))}.groupByKey().flatMap{
//      case( a, b)=>  //b=Interable[(item2,score)]
//        val topk= b.toArray.sortWith{ (a,b) => a._2>b._2 }.take(k)
//        topk.map{ t => (a,t._1,t._2) }
//    }

    //return
    sim.map(t=>ItemSimi(t._1,t._2,t._3))
  }

}
