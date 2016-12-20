package LDA

import java.io.{BufferedWriter, OutputStreamWriter}

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.rdd.RDD
import org.apache.spark.{Logging, SparkContext}

import scala.collection.mutable.StringBuilder

object Writer extends Logging
{
  def sovlePhi(nkv: Array[Array[Int]], 
               nk: Array[Int],
               beta: Double): 
               Array[Array[Double]] = {
      val kTopic = nk.length
      val vSize = nkv.head.length
      val topicTermProb = new Array[Array[Double]](kTopic)

      for (k <- 0 until kTopic) {
        topicTermProb(k) = new Array[Double](vSize)
      }
      for (k <- 0 until kTopic) {
        for (v <- 0 until vSize) {
          topicTermProb(k)(v) = (nkv(k)(v) + beta) / (nk(k) + vSize * beta)
        }
      }
    
      topicTermProb
    }

  def TopicTermParameter( nkv: Array[Array[Int]],
                          nk: Array[Int],
                          beta: Double,
                          sc: SparkContext,
                          output: String) {
      val conf = new Configuration()
      val fs = FileSystem.get(conf)
      val outputPhi = output + "/Phi"
      val br = new BufferedWriter(new OutputStreamWriter(fs.create(new Path(outputPhi),true)));
      val phi = sovlePhi(nkv,nk,beta)

      val serializePhi = phi.zipWithIndex.map{ t => 
        val prob  = t._1
        val index = t._2
        var probabilityLine = new StringBuilder(prob.length, index.toString)

        for ( v <- 0 until prob.length) {
          probabilityLine.append(" ").append(prob(v))
        }
        probabilityLine.append("\n")

        probabilityLine.toString
        br.write(probabilityLine.toString)
      }
      br.close()
    }
  
  def sovleTheta(Documents: RDD[(Int, Array[(Int, Int)],Array[Int])],
                 alpha: Double): 
                 RDD[(Int,Array[Double])] = {
      Documents.map { case (docId, topicAssignArr, nmk) =>
        val kTopic = nmk.length
        val docTopicProb = new Array[Double](kTopic)
        val topicSum = nmk.sum
        
        for (k <- 0 until kTopic) {
          docTopicProb(k) = (nmk(k) + alpha) / (topicSum + kTopic * alpha)
        }
      
        (docId,docTopicProb)
      }
    }

  def DocTopicParameter(Docoments: RDD[(Int, Array[(Int, Int)],Array[Int])],
                        alpha: Double,
                        output: String) {
      val outputTheta = output + "/Theta"
      val DocTopic = sovleTheta(Docoments,alpha)

      DocTopic.map { t => 
        val docId = t._1
        val prob = t._2
        var outputLine = new StringBuilder(prob.length, docId.toString)

        for ( k <- 0 until prob.length) {
          outputLine.append(" ").append(prob(k))
        }
        outputLine.toString
      }.saveAsTextFile(outputTheta)
    }
}
