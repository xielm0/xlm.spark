
/**
 * Created by xieliming on 2016/4/21.
 */
package com.jd.szad.catecf

import com.jd.szad.tools.Writer
import org.apache.spark.{SparkConf, SparkContext}

/*
   &
*/

object CateCF {
  def main(args:Array[String]) {
    val conf = new SparkConf().setAppName("CateCF")
      .set("spark.akka.timeout", "1000")
      .set("spark.driver.maxResultSize", "8g")

    val sc = new SparkContext(conf)

    val input_3: String = "app.db/app_szad_m_dmp_catecf_train/user_type=1/item_type=3/*.lzo"
    compute_sim(sc,input_3,3)

    val input_2: String = "app.db/app_szad_m_dmp_catecf_train/user_type=1/item_type=2/*.lzo"
    compute_sim(sc,input_2,2)

  }

  /*分别计算每个user的(item,item)相似性的分子
  输入x=(user,(item1,item2,...))
  * */
  def compute_numerator(item_seq :Iterable[String] )={
    val item_cnt = item_seq.map(t=> 1).reduce(_+_)
    for (item1 <- item_seq;
         item2<- item_seq;
         if item1 != item2)
      yield { ((item1,item2),1/math.log(1 + item_cnt))}
  }

  def compute_sim(sc :SparkContext ,input_path :String , item_type:Int )={
    //val input: String = "app.db/app_szad_m_dmp_catecf_train/user_type=1/item_type=1/000000_0.lzo"
    val input: String = input_path
    val data=sc.textFile(input).repartition(200)

    val user_item=data.map( _.split("\t") match{ case Array(user,item,rate) =>(user,item)}).cache()

    val item_freq = user_item.map{ case (user,item)=>(item,1) }.reduceByKey(_+_).collect().toMap

    // return (user,Array(item1,item2,...))
    val user_vectors = user_item.groupByKey()

    //compute sim
    val sim_numerator = user_vectors.flatMap{x=> compute_numerator(x._2)}.reduceByKey(_+_).cache()
    print ( " sim_numerator cnt  = " + sim_numerator.first().toString())

    //compute item pair interaction
    val item_pair_join = sim_numerator.map(t=>(t._1,1)).reduceByKey(_+_)


    //计算(item,item)相似性计算的分母，即并集.返回 ((item_1,item2),score)
    //并集=item1集合 + item2集合 - 交集
    val sim_denominator=item_pair_join.map{t=>
      val item1=t._1._1
      val item2=t._1._2
      val join_uv=t._2
      (t._1,item_freq(item1) + item_freq(item2) - join_uv)
    }

    //相似度
    val similary= sim_numerator.join(sim_denominator).map{
      case((item1,item2),(num,denom)) => (item1,item2 ,math.round(10000.0 * num/denom)/10000.0 )}.filter(t=>t._3>0)

    //top 100
    val similary_topk = similary.map{case (item1,item2,score) => (item1,(item2,score))}.groupByKey().flatMap{
      case ( a, b)=>  //b=Interable[(item2,score)]
        val topk= b.toArray.sortWith{case(a,b) => a._2>b._2 }.take(100)
        topk.map{case(b,score) => (a,b,score) }
    }.repartition(10).map(t=>t._1 +"\t" + t._2 +"\t" + t._3  )

    //插入数据库
    Writer.write_table(similary_topk,"app.db/app_szad_m_dmp_cate_cor/item_type="+item_type,"lzo")
  }

}
