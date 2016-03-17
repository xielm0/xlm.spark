package com.jd.spark.ml

import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Random

object GibbsLDA 
{
  //n,代表计数，m,文档总数 k,主题数, v 词典大小，
  //返回：word_topic 一个向量
  def gibbsSampling(word_topic: Array[(Int, Int)],  //词对应的主题(word,topic)，原本应该是NxK矩阵，这里只记录最后的状态
                    nmk: Array[Int],            //原本是一个mxk矩阵，文档X主题，文档对应的主题,这里由于spark使用map,而矩阵的每个元素都是向量。所以这里用向量就好
                    nkv: Array[Array[Int]],     //KxV矩阵， 主题X词
                    nk: Array[Int],             //K向量，主题计数
                    kTopic: Int,
                    vSize: Int,
                    alpha: Double,
                    beta: Double): 
                    (Array[(Int, Int)], Array[Int]) = {
    val length = word_topic.length
    val p = new Array[Double](kTopic)   //记录每次gibss抽样的k主题概率

    var i = 0
    while (i < length)
    {
      //剔除第i个word的影响。
      val topic = word_topic(i)._2
      val word = word_topic(i)._1
      nmk(topic) = nmk(topic) - 1
      nkv(topic)(word) = nkv(topic)(word) - 1
      nk(topic) = nk(topic) - 1
      //gibbs核心计算公式
      //计算每个主题的概率，返回p[k]向量。用p[k]作为多样式分布的概率分布
      for ( k <- 0 until kTopic)
      {
        p(k) = (nmk(k).toDouble + alpha) * (nkv(k)(word) + beta) / (nk(k) + vSize * beta)
      }
      //进行多样式抽样，得到新主题k，并作为Gibbs抽样的返回值。
      val newTopic = Sampler.Multinomial(p)

      word_topic(i) = (word, newTopic)

      nmk(newTopic) = nmk(newTopic) + 1
      nkv(newTopic)(word) = nkv(newTopic)(word) + 1
      nk(newTopic) = nk(newTopic) + 1
      i += 1
    }
    (word_topic, nmk)
  }
  //nkv ,   two-dim array=matrix ,=(topic,word,cnt)
  //nk  , =topic,cnt
  //wordsTopicReduced   (word,topic),
  def updateNKV(wordsTopicReduced: Array[((Int, Int), Int)],
                kTopic: Int,
                vSize: Int): 
                (Array[Array[Int]],Array[Int]) = {
    val nkv = new Array[Array[Int]](kTopic)
    val nk = new Array[Int](kTopic)
    for (k <- 0 until kTopic) 
    {
      nkv(k) = new Array[Int](vSize)
    }
    wordsTopicReduced.foreach
    { 
      t =>

      val word = t._1._1
      val topic = t._1._2
      val count = t._2

      nkv(topic)(word) = nkv(topic)(word) + count
      nk(topic) = nk(topic) + count
    }
    (nkv, nk)
  }
  
  def RunLda( filename: String, 
              maxIter: Int, 
             // blockNums: Int,
              kTopic: Int,
              alpha: Double,
              beta: Double,
              output: String,
              sc: SparkContext) 
  {
    //val (docGrouped,vSize) = Parser.loadCorpus(sc, filename, blockNums)
    val (docGrouped,vSize) = Parser.load_Data(sc, filename)
    //val reduceBlockNum = blockNums / 2

    val kTopicGlobal = sc.broadcast(kTopic)
    val vSizeGlobal  = sc.broadcast(vSize)
    val alphaGlobal  = sc.broadcast(alpha)
    val betaGlobal   = sc.broadcast(beta)

    /*step1 初始时，随机的给训练语料中的每一个词w赋值一个主题z,并统计两个频率计数矩阵：
      Doc-Topic计数矩阵  Ndt,描述每个文档中的主题频率分布
      Word-Topic计数矩阵 Ntw,表示每个主题下词的频率分布  ，k* vSize
      */
    def wt_sum(wt:Array[(Int, Int)]):Array[((Int, Int), Int)]={
      sc.parallelize(wt).map(t=>(t,1)).reduceByKey(_+_).collect()
    }

    //documents=(docId, word_topic, nmk)
    var documents = docGrouped.map { 
      case Document(docId, content) =>

      val length = content.length
      val word_topic = new Array[(Int, Int)](length)  //(word,topic)
      val nmk = new Array[Int](kTopicGlobal.value) //nmk 一个文档的主题频率分布
      val rand = new Random(System.currentTimeMillis/10000)  // rand.setSeed(System.currentTimeMillis/10000)

      for (i <- 0 until length) 
      {
        //val topic = Sampler.Uniform(kTopicGlobal.value, rand)
        val topic =rand.nextInt(kTopicGlobal.value)
        word_topic(i) = (content(i), topic) //random topic
        nmk(topic) = nmk(topic) + 1
      }
      (docId, word_topic, nmk)
      //(docId, wt_sum(word_topic),nmk  )
    }.cache()

    //wordsTopicReduced=得到 word-topic的汇总得分,表示每个主题下词的频率分布,(word-topic),sum
    var wordsTopicReduced = documents.
                            flatMap(t => t._2).
                            map(t => (t, 1)).
                            reduceByKey(_ + _).
                            collect()
    //nkv, two-dim array,=matrix , =topic,word,cnt
    //Step2: 遍历训练语料，按照概率重新采样其中每一个词w对应的主题z，同步更新Nwt和Ntd。
    var (nkv,nk) = updateNKV(wordsTopicReduced, kTopic, vSize)

    var nkvGlobal = sc.broadcast(nkv)
    var nkGlobal  = sc.broadcast(nk)

    //use gibbs sampling to infer the topic distribution in doc and estimate the parameter
    //Step3: 重复 step2，直到Nwt收敛。
    for (iter <- 0 until maxIter) 
    {
      val previousDocuments = documents
      documents = previousDocuments.map 
      { 
        case (docId, word_topic, nmk) =>
        //gibbs sampling
        val (word_topic, newNmk) = gibbsSampling(word_topic, nmk,
                          nkvGlobal.value, nkGlobal.value, kTopicGlobal.value, 
                          vSizeGlobal.value, alphaGlobal.value, betaGlobal.value)

        (docId, word_topic, newNmk)
      }.cache()  

      wordsTopicReduced = documents.
                          flatMap(t => t._2).
                          map(t => (t, 1)).
                          reduceByKey(_ + _).
                          collect()

      //collect topic-term matrix
      var (newnkv,newnk) = updateNKV(wordsTopicReduced, kTopic, vSize)

      //broadcast the global data(shared topic-term matrix)
      nkvGlobal = sc.broadcast(newnkv)
      nkGlobal  = sc.broadcast(newnk)

      if ((iter + 1) == maxIter) {
        Writer.DocTopicParameter(documents, alpha, output)
        Writer.TopicTermParameter(newnkv, newnk, beta, sc, output)
      }

      previousDocuments.unpersist()

      if ((iter + 1) % 20 == 0) {
        documents.saveAsObjectFile(output + "/" + iter.toString)
      }
      if ((iter + 1) % 10 == 0) {
        documents.checkpoint()
      }
    }
  }

  /*
   spark-submit --master yarn-client --class com.jd.spark.ml.GibbsLDA \
   --num-executors 50 --executor-memory 20g \
   /home/jd_ad/xlm/LDA.jar 20 18 app.db/app_szad_m_dmp_label_intrest_t2/entrance_type=1 app.db/xlm_test_mllib/t0
   */
  def main(args: Array[String]) {
    val maxIterations = args(0).toInt
    val kTopic = args(1).toInt
    //val scMaster = args(2)
    val fileName = args(2)
    val output=args(3)
    //val output = "app.db/xlm_test_mllib"
    //val blockNums = args(4).toInt //50

    val alpha = 0.45
    val beta = 0.01

    val conf = new SparkConf()
                   //.setMaster(scMaster)
                   .setAppName("GibbsLDA")
                   .set("spark.akka.timeout", "500")
    val checkPointDir = System.getProperty("spark.gibbsSampling.checkPointDir", output)
    var sc = new SparkContext(conf)
    sc.setCheckpointDir(checkPointDir)

    RunLda(fileName, maxIterations, /*blockNums, */ kTopic, alpha, beta, output, sc)
  }
}
