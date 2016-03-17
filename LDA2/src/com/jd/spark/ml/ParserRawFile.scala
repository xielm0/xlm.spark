package com.jd.spark.ml

import org.apache.spark.rdd.RDD
import org.apache.spark.{HashPartitioner, Logging, Partitioner, SparkContext}

object Parser extends Logging
{
  def parse_data(content:String):Array[Int]={
    val res=content.split(":")
    val cnt=res(1).toInt
    val cate=res(0).toInt
    var a=new Array[Int](cnt)
    for (i<- 0 until   cnt){
      a(i)=cate
    }
    a
  }

  def loadCorpus( sc: SparkContext, 
                  rawfile: String, 
                  numBlocks: Int): 
                  (RDD[Document], Int) = {

    val rawFiles = sc.textFile(rawfile, numBlocks).map { 
      line =>        

      val vs = line.split("\t")
      val words = vs(2).split(",").flatMap(parse_data ).toList
      Document(vs(1).toInt, words)
    }

    val partitioner = new Partitioner {
      val numPartitions = numBlocks

      def getPartition(x: Any): Int = 
      {
        x.asInstanceOf[Int] % numPartitions
      }
    }

    val DocByDocIdBlock = rawFiles.map {
      d => (partitioner.getPartition(d.docId), d)
    }

    val docGrouped = DocByDocIdBlock.
                     partitionBy(new HashPartitioner(numBlocks)).
                     map {t => t._2}

    val vSize = docGrouped.
                flatMap{t => t.content}.
                distinct().
                count().
                toInt

    println("corpus size:")
    println(vSize)

    (docGrouped, vSize)
  }
}
