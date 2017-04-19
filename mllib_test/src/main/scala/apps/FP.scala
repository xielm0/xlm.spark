package apps

import org.apache.spark.{SparkContext, SparkConf}

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.fpm.FPGrowth

/**
 * Created by xieliming on 2017/3/16.
 */
object FP {
  def main(args:Array[String]) {
    val sparkConf = new SparkConf()
      .setAppName("DecisionTree")
    val sc = new SparkContext(sparkConf)

    val data = sc.textFile("data/mllib/sample_fpgrowth.txt")

    val transactions: RDD[Array[String]] = data.map(s => s.trim.split(' '))

    val fpg = new FPGrowth()
      .setMinSupport(0.2)
      .setNumPartitions(10)
    val model = fpg.run(transactions)

    model.freqItemsets.collect().foreach { itemset =>
      println(itemset.items.mkString("[", ",", "]") + ", " + itemset.freq)
    }

    val minConfidence = 0.8
    model.generateAssociationRules(minConfidence).collect().foreach { rule =>
      println(
        rule.antecedent.mkString("[", ",", "]")
          + " => " + rule.consequent.mkString("[", ",", "]")
          + ", " + rule.confidence)
    }
  }

}
