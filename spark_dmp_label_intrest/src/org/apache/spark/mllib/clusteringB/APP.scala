package org.apache.spark.mllib.clusteringB

import breeze.linalg.normalize
import breeze.numerics._
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

//import org.apache.spark.mllib.clustering.{LDA, DistributedLDAModel}


/**
 * Created by xieliming on 2015/11/9.
 */
object APP {

  /*
   nohup spark-submit --master yarn-client --class org.apache.spark.mllib.clusteringB.APP \
   --num-executors 20 --executor-memory 15g --executor-cores 4 \
   /home/jd_ad/xlm/LDA15.jar  \
   app.db/app_szad_m_dmp_label_intrest_t2/entrance_type=1  22 38 1.3 1.05  train 300  &
    */
  //hdfs dfs -rm -r  app.db/app_szad_m_dmp_label_intrest_mode/*

  def main(args: Array[String]): Unit = {
    val filePath = args(0)
    val kTopic = args(1).toInt
    val maxIterations = args(2).toInt
    val alpha = args(3).toDouble  //(50/k)+1
    val beta = args(4).toDouble   //1.1
    val mtype = args(5)
    val num_partitions=args(6).toInt
    val ModelPath = "app.db/app_szad_m_dmp_label_intrest_mode"
    //val tab_partition = args(8)

    //sparkConf
    val sparkConf = new SparkConf().setAppName("SparkLDA").set("spark.rpc.askTimeout", "1000").set("spark.locality.wait","5000")
    val sc = new SparkContext(sparkConf)

    //删除hdfs上的checkpoint
    val conf = new Configuration()
    val fs = FileSystem.get(conf)
    //fs.delete是一个非常危险的动作，所以把它写死。
    fs.delete(new Path( "app.db/app_szad_m_dmp_label_intrest_mode/checkPointDir"),true)
    val checkPointDir = System.getProperty("checkPointDir", ModelPath + "/" + "checkPointDir")
    sc.setCheckpointDir(checkPointDir)

    //写hdfs的用法
    //val br = new BufferedWriter(new OutputStreamWriter(fs.create(new Path(outputPhi),true)))  //先放在流
    //br.write(probabilityLine.toString)  //写入

    //读取类目，并index
    //val vocabRDD = sc.textFile("hdfs://ns1/user/dd_edw/warehouse/dim/dim_item_gen_second_cate")
    val vocabRDD = sc.textFile("hdfs://ns3/user/jd_ad/app.db/app_szad_m_dmp_label_intrest_cate")
  /*  val vocab = vocabRDD.map { t =>
      val cols = t.split("\t")
      (cols(0).toInt, cols(6).toInt)
    }*/

    val vocab1=vocabRDD.map{t=>
       val cols=t.split("\t")
      cols(0).toInt}
    val vocabSize = vocab1.count.toInt //1086
    val vocab2 = vocab1.zipWithIndex().collect.toMap //变成了1个map

    val vocabArray = vocab1.collect

      //读取数据
      //file format： gdt_openid,user_id,cateCnt
      //val textRDD= sc.textFile("app.db/app_szad_m_dmp_label_intrest_test/entrance_type=1")
      val textRDD = sc.textFile(filePath)
      val parsedData = textRDD.repartition(num_partitions).map {
        line =>
          // val line=textRDD.first
          val items = line.split("\t")
          val user_id = items(1).toLong
          val cates = items(2).split(",").map { cateCnt =>
            val s = cateCnt.split(":")
            (vocab2(s(0).toInt).toInt, s(1).toDouble) //return (1,22.0)
          }
          val s2: Vector = Vectors.sparse(vocabSize, cates)
          (user_id, s2) //return
      }

      val corpus = parsedData.cache()

    if (mtype == "train") {
      //删除mode文件
      fs.delete(new Path( "app.db/app_szad_m_dmp_label_intrest_mode/data"),true)
      fs.delete(new Path( "app.db/app_szad_m_dmp_label_intrest_mode/metadata"),true)
      // new LDA
      //val lda=new LDA().setK(10).setMaxIterations(20).setOptimizer("em").setDocConcentration(1.3).setTopicConcentration(1.1)
      val lda = new LDA().setK(kTopic)
        .setMaxIterations(maxIterations)
        .setOptimizer("em")
        .setDocConcentration(alpha)
        .setTopicConcentration(beta)
        .setCheckpointInterval(10)

      val ldaModel = lda.run(corpus) //return LDAModel


      //val topicsM=ldaModel.topicsMatrix
      // describeTopics  每个主题的词权重排序  //return Array[(Array[Int], Array[Double])]
      val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 7) //每个主题取前8个词

/*      val topics = topicIndices.map { case (terms, termWeights) =>
        terms.zip(termWeights).map { case (term, weight) => (vocabArray(term.toInt), weight) }
      }
*/
      //将结果保存在表里 topic  cate1 score  cate2 score  ... cate5 score5
      val top2= topicIndices.map { case (terms, termWeights) =>
        terms.zip(termWeights).flatMap { case (term, weight) =>  Array(vocabArray(term.toInt).toString,  (math.round(weight*100)/100.0).toString  )  }
      }

      val topics2=top2.zipWithIndex.map(_.swap).map{t=>
        (t._1,t._2(0),t._2(1),t._2(2),t._2(3),t._2(4),t._2(5),t._2(6),t._2(7),t._2(8),t._2(9) )
         }
      //topicDistributions   返回训练文档的主题分布概率
      val distLDAModel = ldaModel.asInstanceOf[DistributedLDAModel] // 转换为子类 DistributedLDAModel
      val doc = distLDAModel.topicDistributions

      println("parameter estimates: logLikelihood="+distLDAModel.logLikelihood)
      println("compute prior log probability: logPrior=" +distLDAModel.logPrior)
      println("trainning is ok")

      //取前2个主题作为兴趣
      val instrest = distLDAModel.topTopicsPerDocument(2).map{
        //case (docID, topIndices, weights) =>
        t=>
          (t._1,t._2.mkString(","))
      }
      /*val instrest = distLDAModel.topTopicsPerDocument(5).map {
          //case (docID, topIndices, weights) =>
          t=>
          val w = t._3.filter(_ > 0.2)
          val tt = t._2.slice(0, w.length)
          (t._1, tt.mkString(","))
      }*/


      //将模型结果保存，供下次调用
      //distLDAModel.save(sc,"hdfs://ns3/user/jd_ad/app.db/app_szad_m_dmp_label_intrest_mode")
      distLDAModel.save(sc, ModelPath)
      //将user归属主题插入hive,
      val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)
      import hiveContext.implicits._

      hiveContext.sql("use app")
      //val doc2=doc.map(t=>(t._1,t._2.toArray.mkString(","))).toDF("doc_id","topic")
      val instrest2 = instrest.toDF("doc_id", "topic")
      instrest2.registerTempTable("table1")

      // 将主题表述插入表
      val topicintable=sc.parallelize(topics2).toDF("topic_id","word1","score1","word2","score2","word3","score3","word4","score4","word5","score5")
      topicintable.registerTempTable("topic_table")
      val sql_text2 = "insert overwrite table app_szad_m_dmp_label_intrest_topic partition(dp='ACTIVE',dt='4712-12-31' )  " +
        "select  topic_id ,word1, score1,word2, score2,word3, score3,word4, score4,word5, score5 from topic_table"
      hiveContext.sql(sql_text2)

      //插入文档归属主题，数据量过大，停5s  //暂未找到方法

      val sql_text = "insert overwrite table app_szad_m_dmp_label_intrest_t4 partition(entrance_type=1)  select  null ,doc_id, topic from table1"
      hiveContext.sql(sql_text)

    }else if(mtype=="predict")
    {
      val lda=DistributedLDAModel.SaveLoadV1_0.load2(sc,ModelPath)
      //通过load能获得模型的参数

      def predict1(documents: RDD[(Long, Vector)]): RDD[(Long, Vector)]= {
        val expElogbeta = exp(LDAUtils.dirichletExpectation(lda.topicsMatrix.toBreeze.toDenseMatrix.t).t)
        val expElogbetaBc = corpus.sparkContext.broadcast(expElogbeta)
        val docConcentrationBrz = lda.docConcentration //.toBreeze
        val gammaShape = 100
        val k = lda.k

         documents.map { case (id: Long, termCounts: Vector) =>
          if (termCounts.size == 0) {
            (id, Vectors.zeros(k))
          } else {
            val (gamma, _) = OnlineLDAOptimizer.variationalTopicInference(
              termCounts,
              expElogbetaBc.value,
              docConcentrationBrz,
              gammaShape,
              k)
            (id, Vectors.dense(normalize(gamma, 1.0).toArray))
          }
        }
      }

      //对新的数据进行计算,最简单的方法，p(t|d)=sum(p(w|t))
      //def predict2(documents: RDD[(Long, Vector)]): RDD[(Long, Vector)]

      val res=predict1(corpus)
      //println(res.first)
    }

    sc.stop()

  }


}
