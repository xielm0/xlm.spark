package com.jd.spark.ml

import org.apache.spark.rdd.RDD
import org.apache.spark.{HashPartitioner, Logging, Partitioner, SparkContext}

object Parser extends Logging
{
  //loadCorpus 只是将rawfile进行了hash处理而已
  def loadCorpus( sc: SparkContext, 
                  rawfile: String, 
                  numBlocks: Int): 
                  (RDD[Document], Int) = {

    val rawFiles = sc.textFile(rawfile, numBlocks).map { 
      line =>
      val vs = line.split("\t")
      val words = vs.tail.map(term => term.toInt).toList
      //Document(vs(0).toInt, words)
      Document(vs(0).toInt, words)
    }
    // 通过numblocks进行切片
    //Partitioner 是个抽象类
    val partitioner = new Partitioner {
      //自定义分区个数
      val numPartitions = numBlocks

      //得到x在那个分区
      //
      def getPartition(x: Any): Int = 
      {
        x.asInstanceOf[Int] % numPartitions
      }
    }
    //partitioner.getPartition(d.docId)=docid在那个分区，返回int
    //DocByDocIdBlock 返回 (part id ,分档对象（docId,content） )
    val DocByDocIdBlock = rawFiles.map {
      d => (partitioner.getPartition(d.docId), d)
    }
    //按照文档编号   hash到每个分区里面
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

    //return
    (docGrouped, vSize)
  }
  //将1435:2, 转换成array(1435,1435)
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

  def load_Data(sc: SparkContext,
                rawfile: String
                ): (RDD[Document], Int) = {
    val org_data = sc.textFile(rawfile).repartition(50)
    val rawFiles= org_data.map {
      line=>
        //val line="abc"+"\t"+"100"+"\t"+"123:3,999:2"
        val items = line.split("\t")
        val user = items(1).toInt
        val cates=items(2).split(",").flatMap(parse_data )
          Document(user,cates.toList)
    }
    val docGrouped = rawFiles.cache()
    val vSize = docGrouped.flatMap{t=>t.content}.distinct().count().toInt
    println("corpus size:")
    println(vSize)
    //return
    (docGrouped, vSize)

  }

  //
}
