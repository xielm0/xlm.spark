
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
  def compute_numerator(item_seq :Iterable[Int] )={
    val item_cnt = item_seq.map(t=> 1).reduce(_+_)
    for (item1 <- item_seq;
         item2<- item_seq;
         if item1 < item2)  //可以改为<号，节省一半的空间。
      yield { ((item1,item2),1.0/math.log(1 + item_cnt))}
  }

  def compute_sim( sc:SparkContext ,user_action :RDD[UserItem] ,part_num:Int  ,k :Int): RDD[ItemSimi] ={

    //为了节省内存，将item由字符串转换为int型。
    val item_map = user_action .map (t=> t.itemid).distinct(part_num).zipWithIndex().map(t=> (t._1,t._2.toInt)).cache()
    print("item count is " +item_map.count() )

    val user_item = user_action.map(t=>(t.itemid,t.userid))
      .join(item_map)
      .map{case (item,(user,item_num)) => (user,item_num)}

    val item_freq = user_item.map{t=>(t._2,1)}.reduceByKey(_+_)
    val item_freq_b=sc.broadcast(item_freq)

    val item_pair = user_item.groupByKey().flatMap{x=> compute_numerator(x._2)}.persist(StorageLevel.DISK_ONLY )
    //相似性结算的分子
    val sim_numerator = item_pair.reduceByKey(_+_)
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
    //对极小值做一个图灵估计
    val sim1= sim_numerator.join(sim_denominator).map{
      case((item1,item2),(num,denom)) =>
        val cor = if (num/denom>0.001) math.round(1000*num/denom)/1000.0 else 0.001
        (item1,item2 ,cor )
    }

    val sim =sim1.map{case (item1,item2,score)=> (item2,item1,score)}.union(sim1)

    //top k
    val sim_topk = sim.map{case (item1,item2,score) => (item1,(item2,score))}.groupByKey().flatMap{
      case( a, b)=>  //b=Interable[(item2,score)]
        val topk= b.toArray.sortWith{ (a,b) => a._2>b._2 }.take(k)
        topk.map{ t => (a,t._1,t._2) }
    }


    //转换回Long
    val similary_res = sim_topk.map(t=> (t._1,(t._2,t._3)))
      .join(item_map.map(_.swap))
      .map{case (id1,((id2,score),item1)) => (id2,(item1,score))}
      .join(item_map.map(_.swap))
      .map{case (id2,((item1,score),item2)) => (item2,item1,score)}

    //return
    similary_res.map(t=>ItemSimi(t._1,t._2,t._3))
  }

}
