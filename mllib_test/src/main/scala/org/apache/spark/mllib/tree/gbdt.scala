package org.apache.spark.mllib.tree

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.gbdt.tree.GradientBoostedTrees
import org.apache.spark.mllib.gbdt.tree.configuration.Algo._
import org.apache.spark.mllib.gbdt.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.gbdt.tree.configuration.Strategy
import org.apache.spark.mllib.gbdt.tree.impurity.Gini
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by xieliming on 2016/12/19.
 */
object gbdt {

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

  // Train a GradientBoostedTrees model.
  // The defaultParams for Classification use LogLoss by default.
  val boostingStrategy = BoostingStrategy.defaultParams("Classification")
  val strategy=new Strategy( algo = Classification,
    impurity = Gini, //measure for feature selection
    maxBins = maxBins,
    maxDepth = maxDepth,
    numClasses = 2,
    categoricalFeaturesInfo = categoricalFeatures
  )

  boostingStrategy.setNumIterations(numIterations) //.numIterations = 3 // Note: Use more iterations in practice.
  boostingStrategy.setTreeStrategy(strategy)//.treeStrategy.numClasses = 2


  // model train
  val model = GradientBoostedTrees.train(trainingData.toJavaRDD(), boostingStrategy)


  // Evaluate model on test instances and compute test error
  val PredAndslabel = testData.map { point =>
    val prediction = model.predict(point.features)
    (prediction, point.label)
  }


  PredAndslabel.saveAsTextFile(output)//save prediction score and label


  //evaluate
  val metrics = new BinaryClassificationMetrics(PredAndslabel)

  // ROC Curve
  val roc = metrics.roc

  // AUROC
  val auROC = metrics.areaUnderROC
  println("Area under ROC = " + auROC)

}
