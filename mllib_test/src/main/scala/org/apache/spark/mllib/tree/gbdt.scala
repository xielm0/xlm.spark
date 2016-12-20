package org.apache.spark.mllib.tree

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.{Strategy, BoostingStrategy}
import org.apache.spark.mllib.tree.impurity.Gini
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by xieliming on 2016/12/19.
 */
object gbdt {
  def main(args:Array[String]) {
    val sparkConf = new SparkConf()
      .setAppName("DecisionTree")

    val conf = new SparkConf()
      .setAppName("gbdt")

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

    //parmas
    val numIterations =10

    // Train a GradientBoostedTrees model.
    // The defaultParams for Classification use LogLoss by default.
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    val strategy = new Strategy(algo = Classification,
      impurity = Gini, //measure for feature selection
      maxBins = 16,
      maxDepth = 5,
      numClasses = 2,
      categoricalFeaturesInfo = Map[Int,Int]()
    )

    boostingStrategy.setTreeStrategy(strategy) //.treeStrategy.numClasses = 2
    boostingStrategy.setNumIterations(numIterations) //.numIterations = 3 // Note: Use more iterations in practice.
//    boostingStrategy.setLoss(loss)


    // model train
    val model = GradientBoostedTrees.train(trainData, boostingStrategy)

    // Evaluate model on test instances and compute test error
    val PredAndslabel = testData.map { point =>
      val prediction = model.predict(point.features)
      (prediction, point.label)
    }

    println("PredAndslabel=" + PredAndslabel.collect().toString)


    //evaluate
    val metrics = new BinaryClassificationMetrics(PredAndslabel)

    // ROC Curve
    val roc = metrics.roc

    // AUROC
    val auROC = metrics.areaUnderROC
    println("Area under ROC = " + auROC)
  }

}
