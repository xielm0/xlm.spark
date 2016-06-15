package com.jd.szad.gender

/**
 * Created by wangjing15 on 2016/5/25.
 */

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.{SparkConf, SparkContext}


object gender_model_apply {
   def main(args: Array[String]) {
     //定义变量

     //val ApplyUrl = "app.db/app_szad_m_dmp_label_gender_apply_jdpin/*.lzo"
     val ApplyUrl = args(0)
     println("数据文件路径 = " + ApplyUrl)

     //ModelPath ="app.db/app_szad_m_dmp_label_gender_jdpin_model_save/"
     val ModelPath = args(1)
     println("模型加载路径 = " + ModelPath)

     //ResultPath ="app_szad_m_dmp_label_gender_jdpin_result"
     val ResultPath = args(2)
     println("结果保存表名 = " + ResultPath )

     //dtPartition ="'2016-04"
     val dtPartition = args(3)
     println("保存分区 = " + dtPartition )

     val sparkConf = new SparkConf().setAppName("gender_DT_apply")
     val sc = new SparkContext(sparkConf)

     println("模型加载 " )
     val sameModel = DecisionTreeModel.load(sc, ModelPath)
     //运用模型去预测用户性别（6月有下单的用户）
     val TData = sc.textFile(ApplyUrl)
     val TparsedData = TData.map { line =>
       val xxxx=line.split("\t")
       val user_id = xxxx(0)
       val parts = xxxx.tail.map(_.toDouble)
      (user_id,LabeledPoint(parts(0), Vectors.dense(parts.tail)))
     }
     val TResultAndPreds = TparsedData.map { t =>
       val prediction = sameModel.predict(t._2.features)
       (t._1, prediction)
     }
     //保持预测结果到标签表中
     println("将结果写入hive表中" )
     val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)
     import hiveContext.implicits._
     hiveContext.sql("use app")
     TResultAndPreds.toDF("cl1","cl2").registerTempTable("TableLabel")
     hiveContext.sql("set hive.exec.compress.output=true")
     hiveContext.sql("set mapred.output.compression.codec=com.hadoop.compression.lzo.LzopCodec")
     val exesql = "INSERT overwrite TABLE " + ResultPath +"  partition(dt='"+dtPartition +"') select cl1,cl2 from TableLabel"
     println("Hive执行语句= " + exesql )
     hiveContext.sql(exesql)
     println("完成" )
     sc.stop()

   }

 }
