package CF

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
 * Created by xieliming on 2017/6/15.
 * 计算相似性
 */
class Similary {

  /*分别计算每个user的(item,item)相似性的分子
  输入x=(user,(item1,item2,...))
  * */
  //1个user如果有100个商品，则产生100*100对tuple
  def compute_numerator(item_seq :Iterable[Long] )={
    val item_cnt = item_seq.map(t=> 1).reduce(_+_)
    for (item1 <- item_seq;
         item2<- item_seq;
         if item1 < item2  )
      yield { ((item1,item2),1.0/math.log(1 + item_cnt))}
  }

  def compute_numerator(item_seq :Iterable[Long], ad_sku :Map[Long,Int])={
    val item_cnt = item_seq.map(t=> 1).reduce(_+_)
    for (item1 <- item_seq;
         item2 <- item_seq;
         if item1 != item2 & ad_sku.contains(item2)
    )yield { ((item1,item2),1.0/math.log(1 + item_cnt))}
  }

  // 计算商品相似性
  def itemSimilay( user_item :RDD[UserItem] ): RDD[ItemSimi] ={

    val sc = user_item.sparkContext

    val item_freq = user_item.map{t=>(t.itemid,1)}.reduceByKey(_+_)

    val item_pair = user_item.map(t=>(t.userid,t.itemid)).groupByKey().flatMap{x=>
      compute_numerator(x._2)
    }.persist(StorageLevel.MEMORY_AND_DISK_SER )

    //相似性计算的分子
    val sim_numerator = item_pair.reduceByKey(_+_)
    print ( " sim_numerator cnt  = " + sim_numerator.first().toString())


    //相似性计算的分母，即并集.返回 ((item_1,item2),score)
    //并集=item1集合 + item2集合 - 交集
    val sim_denominator = item_pair.map{ case ((item1,item2),num) =>
      (item1,(item2,num))
    }.join(item_freq).map{case  (item1,((item2,num),fre1)) =>
      (item2,(item1,fre1,num))
    }.join(item_freq).map{case  (item2,((item1,fre1,num),fre2)) =>
      ((item1,item2) , fre1 + fre2 - num)
    }

    //相似度
    val a = 0.001
    val b = 1.0/a
    val sim= sim_numerator.join(sim_denominator).map{
      case((item1,item2),(num,denom)) =>
        val cor = if (num/denom>a )  num/denom else a
        (item1,item2 ,math.round( b * cor)/b )
    }.filter(_._3>a )    //过滤掉极小值

    sim.map(t=>ItemSimi(t._1,t._2,t._3))
  }

  def itemSimilay( user_item :RDD[UserItem]  ,ad_sku :Map[Long,Int] ,sku_cate :RDD[(Long,Int)] ,k :Int): RDD[ItemSimi] ={

    val sc = user_item.sparkContext
    val ad_sku_b = sc.broadcast(ad_sku)

    //为了节省内存，将item由字符串转换为int型。
    //    val item_map = user_action .map (t=> t.itemid).distinct(100).zipWithIndex().map(t=> (t._1,t._2.toInt)).cache()
    //    print("item count is " +item_map.count() )

    val item_freq = user_item.map{t=>(t.itemid,1)}.reduceByKey(_+_)
    //    val item_freq_b=sc.broadcast(item_freq.collect().toMap)

    val item_pair = user_item.map(t=>(t.userid,t.itemid)).groupByKey().flatMap{x=>
      compute_numerator(x._2,ad_sku_b.value)
    }.persist(StorageLevel.MEMORY_AND_DISK_SER )

    //相似性计算的分子
    val sim_numerator = item_pair.reduceByKey(_+_)
    print ( " sim_numerator cnt  = " + sim_numerator.first().toString())

    //计算交集
    //过滤掉不相等的类目
    val item_join = item_pair.map(t=>(t._1,1)).reduceByKey(_+_) .map{case((item1,item2),num) =>
      (item1,(item2,num))
    }.join(sku_cate).map{case(item1,((item2,num),cate1))=>
      (item2,(item1,cate1,num))
    }.join(sku_cate).map { case (item2, ((item1, cate1, num), cate2)) =>
      (item1,item2,num,cate1,cate2)
    }.filter(t=>t._4 == t._5).map(t=>((t._1,t._2),t._3))

    //相似性计算的分母，即并集.返回 ((item_1,item2),score)
    //并集=item1集合 + item2集合 - 交集
    val sim_denominator = item_join.map{ case ((item1,item2),num) =>
      (item1,(item2,num))
    }.join(item_freq).map{case  (item1,((item2,num),fre1)) =>
      (item2,(item1,fre1,num))
    }.join(item_freq).map{case  (item2,((item1,fre1,num),fre2)) =>
      ((item1,item2) , fre1 + fre2 - num)
    }

    //    val sim_denominator=item_join.map{case ((item1,item2),num)=>
    //        val fre1 = item_freq_b.value.get(item1).getOrElse(0)
    //        val fre2 = item_freq_b.value.get(item2).getOrElse(0)
    //      ((item1,item2) , fre1 + fre2 - num)
    //    }

    //相似度
    val a = 0.001
    val b = 1.0/a
    val sim= sim_numerator.join(sim_denominator).map{
      case((item1,item2),(num,denom)) =>
        val cor = if (num/denom>a )  num/denom else a
        (item1,item2 ,math.round( b * cor)/b )
    }.filter(_._3>a )    //过滤掉极小值

    //top k
    //    val sim = sc.textFile("app.db/app_szad_m_dyrec_itemcf_model").map( _.split("\t")) .map(t=>(t(0),t(1).toLong,t(2).toDouble)).filter(_._1=="10645983408")
    val sim_topk = sim.map{case (item1,item2,score) => (item1,(item2,score))}.groupByKey().flatMap{
      case( a, b)=>  //b=Iterable[(item2,score)]
        val topk= b.toArray.sortWith{ (a,b) => a._2>b._2 }.take(k)
        topk.map{ t => (a,t._1,t._2) }
    }

    //return
    sim_topk.map(t=>ItemSimi(t._1,t._2,t._3))
  }
}
