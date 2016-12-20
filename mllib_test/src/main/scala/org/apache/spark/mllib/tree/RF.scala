package org.apache.spark.mllib.tree

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.impurity._
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by xieliming on 2016/12/19.
 */
object RF {
  def main(args:Array[String]) {
    val conf = new SparkConf()
      .setAppName("DecisionTree")
    val sc = new SparkContext(conf)

    val org_data = sc.textFile("app.db/app_szad_m_dmp_label_childmom_train").map(_.split("\t")) //return array[string]

    val data = org_data.map { line =>
        val parts = line.slice(1, 11)
          .map { t: String =>
            if (t == "NULL" || t == "\\N") 0 else t.toDouble
          }
        LabeledPoint(parts(0), Vectors.dense(parts.tail))
      }

    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainData, testData) = (splits(0), splits(1))

    val numClasses = 2
    val impurity:Impurity =Gini  //gini  // variance
    val maxDepth = 5
    val maxBins = 16    //数值分箱数

    val strategy=new Strategy(
      algo = Classification,  // Regression
      impurity = impurity, //measure for feature selection
      maxBins = maxBins,
      maxDepth = maxDepth,
      numClasses = numClasses,
      categoricalFeaturesInfo = Map[Int,Int]()
    )
    //avalid 策略有效性
    strategy.assertValid()

    val rf = new RandomForest(strategy, numTrees = 10, featureSubsetStrategy = "all", seed = 0)

    val rfmodel = rf.run(trainData)

    //predict
    val PredAndslabel = testData.map { point =>
      val prediction = rfmodel.predict(point.features)
      (prediction, point.label)
    }

    //evaluate
    val metrics = new BinaryClassificationMetrics(PredAndslabel)

    // ROC Curve
    val roc = metrics.roc

    // AUROC
    val auROC = metrics.areaUnderROC
    println("Area under ROC = " + auROC)


  }

}
