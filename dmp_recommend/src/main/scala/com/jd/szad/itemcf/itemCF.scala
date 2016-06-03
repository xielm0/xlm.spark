
/**
 * Created by xieliming on 2016/4/21.
 */
package com.jd.szad.itemcf

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

/*
   nohup spark-submit --master yarn-cluster --queue bdp_jmart_adv.bdp_jmart_sz_ad \
   --num-executors 80 --executor-memory 8g --executor-cores 4 \
   --class com.jd.szad.catecf.itemCF \
   /home/jd_ad/xlm/spark_szad_label_xlm.jar  \
   app.db/app_szad_m_dmp_itemcf_train_day/action_type=1 1000  app.db/app_szad_m_dmp_itemcf_res/action_type=1 &

   nohup spark-submit --master yarn-client \
   --num-executors 80 --executor-memory 10g --executor-cores 4 \
   --class com.jd.szad.catecf.itemCF \
   /home/jd_ad/data_dir/xieliming/spark_szad_label_xlm.jar  \
   app.db/app_szad_m_dmp_itemcf_train_day/action_type=5 800  app.db/app_szad_m_dmp_itemcf_res/action_type=5 &

   --conf spark.dynamicAllocation.enabled=true  --conf spark.shuffle.service.enabled=true  --conf spark.dynamicAllocation.maxExecutors=100 \
   --conf "spark.executor.extraJavaOptions=-XX:+UseParallelGC -XX:+UseParallelOldGC"
*/

object itemCF {
  def main(args:Array[String]) {
    val conf = new SparkConf().setAppName("itemCF")
      .set("spark.akka.timeout", "1000")
      .set("spark.rpc.askTimeout", "500")
      .set("spark.storage.memoryFraction","0.1")  //not use cache(),so not need storage memory
    //.set("spark.shuffle.memoryFraction","0.3")

    val input_path :String = args(0)
    val part_num :Int = args(1).toInt
    val output_path:String = args(2)

    val sc = new SparkContext(conf)

    compute_sim(sc,input_path,output_path,part_num)

  }

  /*分别计算每个user的(item,item)相似性的分子
  输入x=(user,(item1,item2,...))
  * */
  //1个user如果有100个商品，则产生100*100对tuple,相当于笛卡尔
  def compute_numerator(item_seq :Iterable[Int] )={
    val item_cnt = item_seq.map(t=> 1).reduce(_+_)
    for (item1 <- item_seq;
         item2<- item_seq;
         if item1 != item2)
      yield { ((item1,item2),1.0/math.log(1 + item_cnt))}
  }

  def compute_numerator2(item_seq :Iterable[Int] )={
    //val item_cnt = item_seq.map(t=> 1).reduce(_+_)
    for (item1 <- item_seq;
         item2<- item_seq;
         if item1 != item2)
      yield { ((item1,item2),1)}
  }

  def compute_sim(sc :SparkContext ,
                  input_path :String ,
                  output_path:String,
                  part_num :Int )={
    //val input: String = "app.db/app_szad_m_dmp_itemcf_train_day/action_type=1/000000_0.lzo"
    val input: String = input_path
    val data=sc.textFile(input).repartition(part_num)

    //为了节省内存，将item由字符串转换为int型。
    val item_map = data.map(_.split("\t")(1)).distinct(100).zipWithIndex().map(t=> (t._1,t._2.toInt)).cache()
    val user_item = data.map( _.split("\t") match{ case Array(user,item,rate) =>(item,user)})
      .join(item_map)
      .map{case (item,(user,item_id)) => (user,item_id)}.persist(StorageLevel.DISK_ONLY )
    //val user_item=data.map( _.split("\t") match{ case Array(user,item,rate) =>(user,item.toLong)}).persist(StorageLevel.DISK_ONLY )

    val item_freq = user_item.map{t=>(t._2,1)}.reduceByKey(_+_)

    // return (user,Array(item1,item2,...))
    val user_vectors = user_item.groupByKey()

    //compute sim
    val sim_numerator = user_vectors.flatMap{x=> compute_numerator(x._2)}.reduceByKey(_+_)
    //val sim_numerator = user_vectors.mapPartitions{ p => p.flatMap(x=> compute_numerator(x._2)).reduce()}

    //compute item pair interaction
    val item_join= user_vectors.flatMap{x=> compute_numerator2(x._2)}.reduceByKey(_+_)

    //计算(item,item)相似性计算的分母，即并集.返回 ((item_1,item2),score)
    //并集=item1集合 + item2集合 - 交集
    val sim_denominator = item_join.map{case ((item1,item2),num) => (item1,(item2,num)) }
      .join(item_freq).map{case  (item1,((item2,num),fre1)) => (item2,(item1,fre1,num))}
      .join(item_freq).map{case  (item2,((item1,fre1,num),fre2)) =>
      ((item1,item2) , fre1 + fre2 - num)
    }
    //相似度
    val sim= sim_numerator.join(sim_denominator).map{
      case((item1,item2),(num,denom)) => (item1,item2 ,num/denom )}

    //top10
    val pair_topk = sim.map{case (item1,item2,score) => (item1,(item2,score))}.groupByKey().flatMap{
      case ( a, b)=>  //b=Interable[(item2,score)]
        val topk= b.toArray.sortWith{case(a,b) => a._2>b._2 }.take(10)
        topk.map{case(b,score) => (a,b,score) }
    }

    //标准化得分 breeze.linalg.normalize
    val value_max= pair_topk.map( _._3 ).max()
    val value_min= pair_topk.map( _._3 ).min()

    val similary = pair_topk.map{case(item1,item2,score)=>(item1,item2,math.round(100.0*(score-value_min)/(value_max - value_min))/100.0 )}.filter(t=>t._3>0)

    //转换回string
    val similary_res = similary.map(t=> (t._1,(t._2,t._3)))
      .join(item_map.map(_.swap))
      .map{case (id1,((id2,score),item1)) => (id2,(item1,score))}
      .join(item_map.map(_.swap))
      .map{case (id2,((item1,score),item2)) => (item2,item1,score)}


    //删除mode文件
    val conf = new Configuration()
    val fs = FileSystem.get(conf)
    fs.delete(new Path( output_path ),true)
    //保存到数据库
    similary_res.map(t=> t._1 +"\t" + t._2 +"\t" + t._3)
      .saveAsTextFile(output_path )


  }

}
