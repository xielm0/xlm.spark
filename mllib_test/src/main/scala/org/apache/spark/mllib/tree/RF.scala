package org.apache.spark.mllib.tree

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

    val numClasses = 2
    val impurity:Impurity =Gini  //gini  // variance
    val maxDepth = 5
    val maxBins = 16    //数值分箱数
    val  categoricalFeatures = Map[Int,Int]()

    val strategy=new Strategy(
      algo = Classification,  // Regression
      impurity = impurity, //measure for feature selection
      maxBins = maxBins,
      maxDepth = maxDepth,
      numClasses = numClasses,
      categoricalFeaturesInfo = categoricalFeatures
    )

    strategy.assertValid()
    val rf = new RandomForest(strategy, numTrees = 1, featureSubsetStrategy = "all", seed = 0)

    val rfModel = rf.run(data)

  }

}
