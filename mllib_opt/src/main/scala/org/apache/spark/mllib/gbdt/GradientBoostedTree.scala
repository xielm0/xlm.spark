package org.apache.spark.mllib.gbdt

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.gbdt.tree.GradientBoostedTrees
import org.apache.spark.mllib.gbdt.tree.configuration.Algo._
import org.apache.spark.mllib.gbdt.tree.configuration.{BoostingStrategy, Strategy}
import org.apache.spark.mllib.gbdt.tree.impurity.Gini
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by shiyouyi on 2016/3/3.
 */
object GradientBoostedTree {

  def main(args:Array[String]) {
    val conf = new SparkConf().setAppName("GBDT")
      .set("spark.akka.timeout", "500")
      .set("spark.driver.maxResultSize", "16g")
    val sc = new SparkContext(conf)

    val input: String = args(0) //input data path，livsvm format
    val maxDepth :Int = args(1).toInt
    val numIterations : Int = args(2).toInt
    val maxBins : Int = args(3).toInt
    val output:String = args(4) // save prediction and label


    val data = MLUtils.loadLibSVMFile(sc,input)

    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

   val  categoricalFeatures = Map[Int,Int]()  // for all features continous

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
    val model = GradientBoostedTrees train(trainingData.toJavaRDD(), boostingStrategy)


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
}
