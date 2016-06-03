
/**
 * Created by xieliming on 2016/4/21.
 */
package com.jd.szad.catecf
import org.apache.spark.{SparkConf, SparkContext}

/*
   nohup spark-submit --master yarn-client \
   nohup spark-submit --master yarn-cluster --queue bdp_jmart_adv.bdp_jmart_sz_ad \
   --num-executors 10 --executor-memory 10g --executor-cores 4 \
   --class com.jd.szad.catecf.CateCF \
   /home/jd_ad/xlm/spark_szad_label_xlm.jar  &
   /home/jd_ad/data_dir/xieliming/spark_szad_label_xlm.jar  &
*/

object CateCF {
  def main(args:Array[String]) {
    val conf = new SparkConf().setAppName("CateCF")
      .set("spark.akka.timeout", "1000")
      .set("spark.driver.maxResultSize", "8g")

    val sc = new SparkContext(conf)

    val input_1: String = "app.db/app_szad_m_dmp_catecf_train/user_type=1/item_type=1/*.lzo"

    compute_sim(sc,input_1,1)

    val input_2: String = "app.db/app_szad_m_dmp_catecf_train/user_type=1/item_type=2/*.lzo"
    compute_sim(sc,input_2,2)

    val input_3: String = "app.db/app_szad_m_dmp_catecf_train/user_type=1/item_type=3/*.lzo"
    compute_sim(sc,input_3,3)

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

  def compute_numerator2(item_seq :Iterable[String] )={
    //val item_cnt = item_seq.map(t=> 1).reduce(_+_)
    for (item1 <- item_seq;
         item2<- item_seq;
         if item1 != item2)
      yield { ((item1,item2),1)}
  }

  def compute_sim(sc :SparkContext ,input_path :String , item_type:Int )={
    //val input: String = "app.db/app_szad_m_dmp_catecf_train/user_type=1/item_type=1/000000_0.lzo"
    val input: String = input_path
    val data=sc.textFile(input)

    val user_item=data.map( _.split("\t") match{ case Array(user,item,rate) =>(user,item)}).cache()

    val item_freq = user_item.map{ case (user,item)=>(item,1) }.reduceByKey(_+_).collect().toMap

    // return (user,Array(item1,item2,...))
    val user_vectors = user_item.groupByKey().cache()

    //compute sim
    val sim_numerator = user_vectors.flatMap{x=> compute_numerator(x._2)}.reduceByKey(_+_)

    //compute item pair interaction
    val item_pair_join=user_vectors.flatMap{x=> compute_numerator2(x._2)}.reduceByKey(_+_)


    //计算(item,item)相似性计算的分母，即并集.返回 ((item_1,item2),score)
    //并集=item1集合 + item2集合 - 交集
    val sim_denominator=item_pair_join.map{t=>
      val item1=t._1._1
      val item2=t._1._2
      val join_uv=t._2
      (t._1,item_freq(item1) + item_freq(item2) - join_uv)
    }

    //相似度
    val sim= sim_numerator.join(sim_denominator).map{
      case((item1,item2),(num,denom)) => (item1,item2 ,num/denom )}

    //标准化得分 breeze.linalg.normalize
    val value_max= sim.map{case(item1,item2,score)=>score}.max()
    val value_min= sim.map{case(item1,item2,score)=>score}.min()

    val similary = sim.map{case(item1,item2,score)=>(item1,item2,math.round(100.0*(score-value_min)/(value_max - value_min))/100.0 )}.filter(t=>t._3>0)

    //top 20
    val similary_topk = similary.map{case (item1,item2,score) => (item1,(item2,score))}.groupByKey().flatMap{
      case ( a, b)=>  //b=Interable[(item2,score)]
        val topk= b.toArray.sortWith{case(a,b) => a._2>b._2 }.take(20)
        topk.map{case(b,score) => (a,b,score) }
    }

    //插入数据库
    val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)
    import hiveContext.implicits._
    val res_df = similary_topk.toDF("item1", "item2", "sim")

    hiveContext.sql("use app")
    res_df.registerTempTable("my_table")
    hiveContext.sql("set hive.exec.compress.output=true")
    hiveContext.sql("set mapred.output.compression.codec=com.hadoop.compression.lzo.LzopCodec")
    val sql_text = "insert overwrite table app_szad_m_dmp_cate_cor partition(item_type= "+ item_type +")  select  item1 ,item2, sim  from my_table"
    hiveContext.sql(sql_text)

  }

}
