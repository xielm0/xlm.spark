package com.jd.szad.intrest

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.mllib.clustering.{DistributedLDAModel, LDA, LDAModel, LocalLDAModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by xieliming on 2016/5/4.
 */

object lda_jd {
  def main(args: Array[String]): Unit = {
    val mtype = args(0)
    val kTopic = args(1).toInt
    val maxIterations = args(2).toInt
    val alpha = args(3).toDouble //(50/k)+1
    val beta = args(4).toDouble //1.1

    val input_path = args(5)
    val num_partitions = args(6).toInt
    val target_path =args(7)
    // val target_path ="app.db/app_szad_m_dmp_label_intrest_res_jdpin"

    val model_path ="app.db/app_szad_m_dmp_label_intrest_mode_jdpin"

    //sparkConf
    val sparkConf = new SparkConf()
      .setAppName("LDA_jdpin")
      .set("spark.akka.timeout", "1000")
      .set("spark.rpc.askTimeout", "500")
      .set("spark.shuffle.memoryFraction","0.3")
      //.set("spark.storage.memoryFraction","0.2")
      //.set("spark.akka.frameSize","128")

    val sc = new SparkContext(sparkConf)

    //删除hdfs上的checkpoint
    val conf = new Configuration()
    val fs = FileSystem.get(conf)
    //fs.delete是一个非常危险的动作，所以把它写死。
    fs.delete(new Path( model_path +"/checkPointDir"), true)
    val checkPointDir = System.getProperty("checkPointDir", model_path +"/checkPointDir")
    sc.setCheckpointDir(checkPointDir)

    //读取类目，并index
    //这个很重要，特别是在predict中，要求是不变的，也就是字典尽量全
    val vocabRDD = sc.textFile("hdfs://ns3/user/jd_ad/app.db/app_szad_m_dmp_label_intrest_cate_jdpin")
    val vocab = vocabRDD.map (_.split("\t")(0)).sortBy(x=>x)

    //词汇表的向量顺序非常重要，不能变，否则预测的时候会出问题。
    val vocabSize:Int = vocab.count.toInt //232
    val vocabMap = sc.broadcast(vocab.zipWithIndex().collect.toMap) //变成了1个map
    val vocabArray = vocab.collect

    //读取数据
    //file format： gdt_openid,user_id,cateCnt
    //val textRDD= sc.textFile("app.db/app_szad_m_dmp_label_intrest_jdpin_train/000000_0.lzo").sample(false,0.05)
    val textRDD = sc.textFile(input_path).repartition(num_partitions)
    //将jdpin转成user_id,int
    val user_rdd= textRDD.map(t=>t.split("\t")(0)).distinct(100).zipWithIndex().persist(StorageLevel.DISK_ONLY )

    val parsedData=textRDD.map{
      line =>
        // val line=textRDD.first
        val items = line.split("\t")
        val jdpin = items(0)
        val cates = items(2).split(",").map { cateCnt =>
          val s = cateCnt.split(":")
          (vocabMap.value(s(0)).toInt, s(1).toDouble) //return (1,22.0)
        }
        val s2 :Vector= Vectors.sparse(vocabSize, cates)
        (jdpin, s2) //return
    }.join(user_rdd).map{case (jdpin,(s2,user_id)) =>(user_id,s2)}

   /* val parsedData = textRDD.map {
      line =>
        // val line=textRDD.first
        val items = line.split("\t")
        val user_id = items(1).toLong
        val cates = items(2).split(",").map { cateCnt =>
          val s = cateCnt.split(":")
          (vocabMap.value(s(0)).toInt, s(1).toDouble) //return (1,22.0)
        }
        val s2 :Vector= Vectors.sparse(vocabSize, cates)
        (user_id, s2) //return
    }*/

    val corpus = parsedData.persist(StorageLevel.DISK_ONLY )//.cache()

    if (mtype == "train") {

      // new LDA
      // val lda=new LDA().setK(10).setMaxIterations(15).setOptimizer("em").setDocConcentration(1.3).setTopicConcentration(1.1)
      val lda = new LDA().setK(kTopic)
        .setMaxIterations(maxIterations)
        .setOptimizer("em")
        .setDocConcentration(alpha)
        .setTopicConcentration(beta)
        .setCheckpointInterval(10)

      val ldaModel = lda.run(corpus)

      //val topics_matrix = ldaModel.topicsMatrix  //this is a matrix of size vocabSize x k, where each column is a topic.

      // describeTopics  每个主题的词权重排序
      val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 7)
      val topicIndices2= topicIndices.map { case (terms, termWeights) =>
        terms.zip(termWeights).flatMap { case (term, weight) =>  Array(vocabArray(term.toInt),  (math.round(weight*100)/100.0).toString  )  }
      }

      //对主题进行编号 topic  cate1 score  cate2 score  ... cate5 score5
      val topics=topicIndices2.zipWithIndex.map(_.swap).map{t=>
        (t._1,t._2(0),t._2(1),t._2(2),t._2(3),t._2(4),t._2(5),t._2(6),t._2(7),t._2(8),t._2(9) )
      }

      //将主题映射保存到hadoop
//      fs.delete(new Path( "app.db/app_szad_m_dmp_label_intrest_topic_jdpin/dp=ACTIVE/dt=4712-12-31/*"),true)
//      sc.parallelize(topics)
//        .map(t=>t._1 +"\t"+t._2 +"\t"+t._3 +"\t"+t._4 +"\t"+t._5 +"\t"+t._6 +"\t"+t._7 +"\t"+t._8 +"\t"+t._9 +"\t"+t._10 +"\t"+t._11)
//        .saveAsTextFile("app.db/app_szad_m_dmp_label_intrest_topic_jdpin/dp=ACTIVE/dt=4712-12-31/")

      //hive
      val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)
      import hiveContext.implicits._

      hiveContext.sql("use app")
      val topic_DF=sc.parallelize(topics).repartition(1).toDF("topic_id","word1","score1","word2","score2","word3","score3","word4","score4","word5","score5")
      topic_DF.registerTempTable("topic_table")

      val sql_text2 = "insert overwrite table app_szad_m_dmp_label_intrest_topic_jdpin partition(dp='ACTIVE',dt='4712-12-31' )  " +
        "select  topic_id ,word1, score1,word2, score2,word3, score3,word4, score4,word5, score5 from topic_table"

      //hiveContext.sql("set hive.exec.compress.output=true")
      //hiveContext.sql("set mapred.output.compression.codec=com.hadoop.compression.lzo.LzopCodec")
      hiveContext.sql(sql_text2)

      //topicDistributions   返回训练文档的主题分布概率
      val distLDAModel = ldaModel.asInstanceOf[DistributedLDAModel]

      println("parameter estimates: logLikelihood="+distLDAModel.logLikelihood)
      println("compute prior log probability: logPrior=" +distLDAModel.logPrior)

      // 将模型结果保存，供下次调用
      //删除mode文件
      fs.delete(new Path( model_path +"/data"),true)
      fs.delete(new Path(  model_path +"/metadata"),true)
      distLDAModel.save(sc, model_path)


    }else if(mtype=="predict"){

      val sameModel =DistributedLDAModel.load(sc,model_path)


      //return An RDD of (document ID, topic mixture distribution for document)
      def predict(documents: RDD[(Long, Vector)], ldaModel: LDAModel): RDD[(Long, Vector)] = {
        ldaModel match {
          case localModel: LocalLDAModel =>
            localModel.topicDistributions(documents)
          case distModel: DistributedLDAModel =>
            distModel.toLocal.topicDistributions(documents)
        }
      }

      val res = predict(corpus,sameModel).join(user_rdd.map(_.swap)).map{case(user_id,(docTopicsWeight,jdpin))=>(jdpin,docTopicsWeight)}

      //取topk , k=2
      val result = res.map{case(doc_id,docTopicsWeight)=>
        val  topic_id= docTopicsWeight.toArray.zipWithIndex.sortWith{  case(a,b)=> a._1>b._1}.map( _._2 )
        (doc_id,topic_id(0).toString + "," + topic_id(1).toString )
      }

      //取Weight>0.15 的topic保存下来。 这个weight不好确定。
 /*     val result = res.map{case(doc_id,docTopicsWeight)=>
        val  topic_id= docTopicsWeight.toArray.zipWithIndex.sortWith{  case(a,b)=> a._1>b._1}.map( _._2 ).filter( _>0.15 )
        (doc_id,topic_id.mkString("\t"))
      }*/

      //保存到hadoop
//      val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)
//      import hiveContext.implicits._
//
//      hiveContext.sql("use app")
//      val instrest = result.toDF("doc_id", "topic")
//      instrest.registerTempTable("table1")
//      val sql_text = "insert overwrite table app_szad_m_dmp_label_intrest_res_jdpin select  doc_id ,null, topic from table1"
//
//      hiveContext.sql("set hive.exec.compress.output=true")
//      hiveContext.sql("set mapred.output.compression.codec=com.hadoop.compression.lzo.LzopCodec")
//      hiveContext.sql(sql_text)

      val conf = new Configuration()
      val fs = FileSystem.get(conf)
      fs.delete(new Path( target_path ),true)

      result.map(t=> t._1 +"\t" + "" +"\t" + t._2)
        .saveAsTextFile(target_path)

    }
    sc.stop()

  }
}
