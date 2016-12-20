package org.apache.spark.mllib.improve

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
}
