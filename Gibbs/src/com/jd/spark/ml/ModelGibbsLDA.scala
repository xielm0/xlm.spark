package com.jd.spark.ml 

import org.apache.spark.SparkContext

object MGibbsLDA 
{
  def gibbsSampling(topicAssignArr: Array[(Int, Int)], 
                    nmk: Array[Int], 
                    nkv: Array[Array[Int]], 
                    nk: Array[Int],
                    kTopic: Int,
                    vSize: Int,
                    alpha: Double,
                    beta: Double): 
                    (Array[(Int, Int)], Array[Int]) = {
    val length = topicAssignArr.length
    val topicDist = new Array[Double](kTopic)

    var i = 0
    while (i < length)
    {
      val topic = topicAssignArr(i)._2
      val word = topicAssignArr(i)._1

      nmk(topic) = nmk(topic) - 1
      nkv(topic)(word) = nkv(topic)(word) - 1
      nk(topic) = nk(topic) - 1
      
      for ( k <- 0 until kTopic)
      {
        topicDist(k) = (nmk(k).toDouble + alpha) * (nkv(k)(word) + beta) / (nk(k) + vSize * beta)
      }
      val newTopic = Sampler.Multinomial(topicDist)
      topicAssignArr(i) = (word, newTopic)

      nmk(newTopic) = nmk(newTopic) + 1
      nkv(newTopic)(word) = nkv(newTopic)(word) + 1
      nk(newTopic) = nk(newTopic) + 1
      i += 1
    }
    (topicAssignArr, nmk)
  }

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
              blockNums: Int,
              kTopic: Int,
              alpha: Double,
              beta: Double,
              output: String,
              sc: SparkContext,
              startPoint: Int) 
  {
    val reduceBlockNum = blockNums / 2
    var documents = sc.
                    objectFile[(Int, Array[(Int, Int)],Array[Int])](filename).
                    cache()
    val vSize = documents.
                flatMap { t => t._2 }.
                map { t => t._1 }.
                distinct().
                count().
                toInt

    val kTopicGlobal = sc.broadcast(kTopic)
    val vSizeGlobal  = sc.broadcast(vSize)
    val alphaGlobal  = sc.broadcast(alpha)
    val betaGlobal   = sc.broadcast(beta)

    //collect count of term-topic(assignment) pairs
    var wordsTopicReduced = documents.
                            flatMap(t => t._2).
                            map(t => (t, 1)).
                            reduceByKey(_ + _, reduceBlockNum).
                            collect()

    var (nkv,nk) = updateNKV(wordsTopicReduced, kTopic, vSize)

    var nkvGlobal = sc.broadcast(nkv)
    var nkGlobal  = sc.broadcast(nk)

    //use gibbs sampling to infer the topic distribution in doc and estimate the parameter
    for (iter <- startPoint until maxIter) 
    {
      val previousDocuments = documents
      documents = previousDocuments.map 
      { 
        case (docId, topicAssignArr, nmk) =>
        //gibbs sampling
        val (newTopicAssignArr, newNmk) = gibbsSampling(topicAssignArr, nmk, 
                          nkvGlobal.value, nkGlobal.value, kTopicGlobal.value, 
                          vSizeGlobal.value, alphaGlobal.value, betaGlobal.value)

        (docId, newTopicAssignArr, newNmk)
      }.cache()  

      wordsTopicReduced = documents.
                          flatMap(t => t._2).
                          map(t => (t, 1)).
                          reduceByKey(_ + _, reduceBlockNum).
                          collect()

      //collect topic-term matrix
      var (newnkv,newnk) = updateNKV(wordsTopicReduced, kTopic, vSize)

      //broadcast the global data(shared topic-term matrix)
      nkvGlobal = sc.broadcast(newnkv)
      nkGlobal  = sc.broadcast(newnk)

      if ((iter + 1) == maxIter) {
        //Writer.DocTopicParameter(documents, alpha, output)
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
  def main(args: Array[String]) {
    val maxIterations = args(0).toInt
    val kTopic = args(1).toInt
    val scMaster = args(2)
    val fileName = args(3)
    val blockNums = args(4).toInt
    val startPoint = args(5).toInt

    val alpha = 0.45
    val beta = 0.01
    val output = "/tmp/data/lda_model_topic2000"

    val conf = new SparkConf()
                   .setMaster(scMaster)
                   .setAppName("GibbsLDA")
                   .set("spark.akka.timeout", "500")
                   .set("spark.akka.askTimeout", "60")
    val checkPointDir = System.getProperty("spark.gibbsSampling.checkPointDir", output)
    var sc = new SparkContext(conf)
    sc.setCheckpointDir(checkPointDir)

    RunLda(fileName, maxIterations, blockNums, kTopic, alpha, beta, output, sc, startPoint)
  }
  */
}
