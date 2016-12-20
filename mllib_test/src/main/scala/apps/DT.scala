package apps

/**
 * Created by xieliming on 2015/11/11.
 */

import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.{SparkConf, SparkContext}

  object DT {
    def main(args:Array[String]) {
      val sparkConf = new SparkConf()
        .setAppName("DecisionTree")
      val sc = new SparkContext(sparkConf)

      val org_data = sc.textFile("app.db/app_szad_m_dmp_label_childmom_train").map(_.split("\t")) //return array[string]

      val data = org_data.map { line =>
        val parts = line.slice(1, 11)
          .map { t: String =>
            if (t == "NULL" || t == "\\N") 0 else t.toDouble
          }
        LabeledPoint(parts(0), Vectors.dense(parts.tail))
      }

      //统计
      data.map(t=>(t.label,1)).reduceByKey(_+_).collect()
//      0.0,107279628
//      1.0,15198988

      //sampling 类似改变先验概率
      val data0 = data.filter(t=>t.label==0).sample(false,0.1)
      val data1 = data.filter(t=>t.label==1).sample(false,0.3)
      val training=data0.union(data1)

      //params
      val numClasses = 2
      val categoricalFeaturesInfo = Map[Int, Int]()
      val impurity = "entropy"  //gini  // variance
      val maxDepth = 5
      val maxBins = 16    //数值分箱数

      val model_DT = DecisionTree.trainClassifier(training, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

      //模型评估
      val PredAndslabel_DT = data.map { point =>
        val prediction = model_DT.predict(point.features)
        (prediction, point.label)
      }

//    要求顺序必须是（预测值, 实际值）
      val matrics_DT=new MulticlassMetrics(PredAndslabel_DT)

      println(matrics_DT.confusionMatrix)
//      9939303.0  786849.0
//      2555005.0  2004121.0

      //precision  ,recall
      println( matrics_DT.precision(1) , matrics_DT.recall(1) )
//      0.7180732863484738  0.4395844729889018

      val b_metrics=new BinaryClassificationMetrics(PredAndslabel_DT)
      // ROC Curve
      val roc = b_metrics.roc

      // AUROC
      val auROC = b_metrics.areaUnderROC
      println("Area under ROC = " + auROC)


      //打印决策树
      model_DT.toDebugString

      //save model
      model_DT.save(sc,"app.db/app_szad_m_dmp_label_childmom_model")

    }
  }


