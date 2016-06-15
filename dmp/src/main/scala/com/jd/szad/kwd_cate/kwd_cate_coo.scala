package com.jd.szad.kwd_cate

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
   * Created by wangjing15 on 2016/5/25.
   */
object kwd_cate_coo {

      def main(args: Array[String]): Unit = {

        //定义变量
        //val FileUrl = "jiangxue/search_cps_ads/"+getNowDate()+"/query_category/"
        //val FileUrl = "jiangxue/search_cps_ads/20160420/query_category/"
        val FileUrl = args(0)
        println("来源文件路径 = " + FileUrl)

        //ResultPath ="app_szad_m_dmp_kwd_cate_coo"
        val ResultPath = args(1)
        println("结果保存表名 = " + ResultPath )

        //dtPartition ="'2016-05-26"
        val dtPartition = args(2)
        println("保存分区 = " + dtPartition )

          val sparkConf = new SparkConf().setAppName("kwd_cate_coo")
          val sc = new SparkContext(sparkConf)

          //加载文件
          val TData = sc.textFile(FileUrl)
          //对入库文件数据进行处理，只取前两个字段
          val TparsedData = TData.map { line =>
            val row = line.split("\t")
            val kwd = row(0)
            val newline=row(1)
            val newlines=newline.replaceAll(" ","\t")
            val newrow = newlines.split("\t")
            val threee_cate = newrow(0)
            (kwd, threee_cate)
          }

          //保持文件前2个字段到表中
          val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)
          import hiveContext.implicits._
          hiveContext.sql("use app")

          TparsedData.toDF("cl1","cl2").registerTempTable("TableLabel")
          hiveContext.sql("set hive.exec.compress.output=true")
          hiveContext.sql("set mapred.output.compression.codec=com.hadoop.compression.lzo.LzopCodec")
        //    val exesql = "INSERT overwrite TABLE app_szad_m_dmp_kwd_cate_coo partition dt='2016-05-26' select cl1,cl2 from TableLabel"
          val exesql = "INSERT overwrite TABLE "+ ResultPath +" partition(dt='"+dtPartition+"') select cl1,cl2 from TableLabel"
          println("Hive执行语句 = " + exesql )

          hiveContext.sql(exesql)
          sc.stop()

      }


  }
