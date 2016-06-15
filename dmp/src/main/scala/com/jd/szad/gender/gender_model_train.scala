package com.jd.szad.gender

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
object gender_model_train {

     def main(args: Array[String]): Unit = {
       //定义变量
       //val TrainUrl = "app.db/app_szad_m_dmp_label_gender_jdpin_train/*.lzo"
       val TrainUrl = args(0)
       println("训练数据文件路径 = " + TrainUrl)

       //ModelPath ="app.db/app_szad_m_dmp_label_gender_jdpin_model_save/"
       val ModelPath = args(1)
       println("模型保存路径 = " + ModelPath)

       val sparkConf = new SparkConf().setAppName("gender_DT")
       val sc = new SparkContext(sparkConf)

         //删除上次建立的模型中的元素 data ，metadata
         val conf = new Configuration()
         val fs = FileSystem.get(conf)
         //fs.delete是一个非常危险的动作，所以把它写死。
         fs.delete(new Path(ModelPath+"data"), true)
         fs.delete(new Path(ModelPath+"metadata"), true)

         //重新建立模型
         //加载文件
         val covTypeData = sc.textFile(TrainUrl)
         val parsedData = covTypeData.map { line =>
           val parts = line.split('\t').slice(1, 56).map(_.toDouble)
           LabeledPoint(parts(0), Vectors.dense(parts.tail))
         }

         /*在初次建立模型时，可采用交叉验证方式对模型参数进行评估*/
         //val Array(train, cvData) = parsedData.randomSplit(Array(0.8, 0.2))
         //train.cache()
         //cvData.cache()
         //evaluate(train,cvData)


         //通过参数评估结果调整，以及人工经验选定的模型参数，建立模型,树的深度不宜过深，防止过拟合,此处设depth=10, bins=300
         val Ymodel = DecisionTree.trainClassifier(parsedData, 3, Map[Int, Int](), "gini", 10, 300)

         //验证模型对样本预测的准确性
         val labelAndPreds2 = parsedData.map { point =>
           val prediction = Ymodel.predict(point.features)
           (point.label, prediction)
         }
         //模型错误率评价
         val trainErr = labelAndPreds2.filter(r => r._1 != r._2).count().toDouble / parsedData.count
         println("Training Error = " + trainErr)

         //打印模型混淆矩阵
         println("模型混淆矩阵")
         val matrics = new MulticlassMetrics(labelAndPreds2)
         println(matrics.confusionMatrix)
         //运行结果
         //    1995414.0  653.0     1426.0
         //    904.0      997063.0  759.0
         //    2399.0     822.0     994351.0
         //查看各分类的准确率和召回率
         (0 until 3).map(
           cat => (matrics.precision(cat), matrics.recall(cat))
         ).foreach(println)
         //运行结果
         //    (0.9983474398826847,0.9989591953513729)
         //    (0.998522840392654,0.9983348786353815)
         //    (0.9978074048504018,0.9967711603773963)

         //对模型结果满意后，可对模型进行保存
         Ymodel.save(sc, ModelPath)
         println("模型保存成功")

       sc.stop()

     }

     /**
      * 模型评估
      * @param trainData 训练数据
      * @param cvData 交叉验证数据
      */
     def evaluate(trainData: RDD[LabeledPoint], cvData: RDD[LabeledPoint]): Unit = {
       val evaluations =
         for (impurity <- Array("gini", "entropy");
              depth <- Array(1, 20);
              bins <- Array(20, 300))
           yield {
             val model = DecisionTree.trainClassifier(
               trainData, 3, Map[Int,Int](), impurity, depth, bins)
             val predictionsAndLabels = cvData.map(example =>
               (model.predict(example.features), example.label)
             )
             val accuracy =
               new MulticlassMetrics(predictionsAndLabels).precision
             ((impurity, depth, bins), accuracy)
           }
       evaluations.sortBy(_._2).reverse.foreach(println)

     }

 }
