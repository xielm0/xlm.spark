package org.apache.spark.mllib.clusteringB

import breeze.linalg.normalize
import breeze.numerics._
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext};

//import org.apache.spark.mllib.clustering.{LDA, DistributedLDAModel}


/**
 * Created by xieliming on 2015/11/9.
 */
object APP {


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

    val sparkConf = new SparkConf()
      .setAppName("SparkLDA")
      .set("spark.akka.timeout", "1000")
      .set("spark.rpc.askTimeout", "500")
      .set("spark.shuffle.memoryFraction","0.3")

    val sc = new SparkContext(sparkConf)

    //删除hdfs上的checkpoint
    val conf = new Configuration()
    val fs = FileSystem.get(conf)
    //fs.delete是一个非常危险的动作，所以把它写死。
    fs.delete(new Path(  ModelPath + "/" + "checkPointDir"),true)

    val checkPointDir = System.getProperty("checkPointDir", ModelPath + "/" + "checkPointDir")
    sc.setCheckpointDir(checkPointDir)

    //读取类目，并index
    val vocabRDD = sc.textFile("hdfs://ns3/user/jd_ad/app.db/app_szad_m_dmp_label_intrest_cate")
  /*  val vocab = vocabRDD.map { t =>
      val cols = t.split("\t")
      (cols(0).toInt, cols(6).toInt)
    }*/

    //val vocab1=vocab.filter(t=>t._2==1).map(t=>t._1)  //只要有效标志
    //val vocab1 = vocab.map(_._1)
    val vocab1=vocabRDD.map{t=>
       val cols=t.split("\t")
      cols(0).toInt}
    val vocabSize = vocab1.count.toInt //1086
    val vocab2 = vocab1.zipWithIndex().collect.toMap //变成了1个map

    val vocabArray = vocab1.collect

      //读取数据
      //格式要求：Array((0,[1.0,2.0,6.0,0.0,2.0,3.0,1.0,1.0,0.0,0.0,3.0]), (1,[1.0,3.0,0.0,1.0,3.0,0.0,0.0,2.0,0.0,0.0,1.0]),...)
      // word是经过排序的
      //file format： gdt_openid,user_id,cateCnt
      //val textRDD= sc.textFile("app.db/app_szad_m_dmp_label_intrest_test/entrance_type=1")
      val parsedData = sc.textFile(filePath).repartition(num_partitions)map {
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
      print("occur action" +corpus.first())

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

      //topicsMatrix: Returns a vocabSize by k matrix where each column is a topic
      //val topicsM=ldaModel.topicsMatrix
      // describeTopics  每个主题的词权重排序  //return Array[(Array[Int], Array[Double])]
      val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 7) //每个主题取前8个词

/*      val topics = topicIndices.map { case (terms, termWeights) =>
        terms.zip(termWeights).map { case (term, weight) => (vocabArray(term.toInt), weight) }
      }
      //打印出来
      topics.zipWithIndex.foreach { case (topic, i) =>
        println(s"TOPIC $i")
        topic.foreach { case (term, weight) =>
          println(s"$term\t$weight")
        }
        println()
      }*/

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

      //将user归属主题插入hive
      val contents = instrest.map(t=> t._1 + "\t" + t._2)
      //Writer.write_table(contents,"app.db/app_szad_m_dmp_label_intrest_t4/entrance_type=1")

      val contents_topic = sc.parallelize(topics2)
        .map(t=> t._1 +"\t"+t._2+"\t"+t._3+"\t"+t._4+"\t"+t._5+"\t"+t._6+"\t"+t._7+"\t"+t._8+"\t"+t._9+"\t"+t._10+"\t"+t._11)
     // Writer.write_table(contents_topic,"app.db/app_szad_m_dmp_label_intrest_topic/dp=ACTIVE/dt=4712-12-31")


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
