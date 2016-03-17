package dm.share

import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by xieliming on 2015/11/12.
 */
object Byes {
  def main(args:Array[String]) {
    val sparkConf = new SparkConf().setAppName("DecisionTree")
    val sc = new SparkContext(sparkConf)

    val data = sc.textFile("app.db/app_szad_m_dmp_label_gender_xlm") //return array[string]

    val parseData = data.map { line =>
      val parts = line.split('\t').map(_.toDouble)
      LabeledPoint(parts(0), Vectors.dense(parts.tail))
    }

    val splits = parseData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training=splits(0).cache()

    val model_byes = NaiveBayes.train(training, lambda = 1.0/*, modelType = "multinomial"*/)
    //multinomial 多项式分布

    //模型评估
    val predict1_byes = training.map(p => (model_byes.predict(p.features),p.label))
    val matrics_byes=new MulticlassMetrics(predict1_byes)
    println(matrics_byes.confusionMatrix)
//    1118299.0  235351.0  501972.0
//    158277.0   370918.0  213746.0
//    744248.0   404899.0  1472941.0
    //计算准确率\召回率
    (0 until 3).map(
      sex => (matrics_byes.precision(sex), matrics_byes.recall(sex))
    ).foreach(println)
//    (0.5533876280170861,0.6026545277001458)
//    (0.3668213392828887,0.4992563339484562)
//    (0.6729878889310761,0.5617435417880712)

    //ROC & AUC
    val roc_metrics=new BinaryClassificationMetrics(predict1_byes)
    println("area under PR:"+roc_metrics.areaUnderPR() +"  AUC:"+roc_metrics.areaUnderROC())
//    area under PR:0.8109519106138725  AUC:0.6678262266955499

    /*  val data2= sc.textFile("app.db/app_szad_f_dmp_gender_xlm_yuce_m/dt=2015-11-01")
    val test = data2.map { line =>
      val parts = line.split('\t').slice(2,17).map(_.toDouble)
      LabeledPoint(parts(0), Vectors.dense(parts.tail))
    }*/
    val test=splits(1)
    val predict2 = test.map(p => (model_byes.predict(p.features), p.label))

    //模型评估
    val matrics2=new MulticlassMetrics(predict2)
    println(matrics2.confusionMatrix)
    //计算准确率\召回率
    (0 until 2).map(
      sex => (matrics2.precision(sex), matrics2.recall(sex))
    ).foreach(println)

    val accuracy = 1.0 * predict2.filter(x => x._1 == x._2).count()/ test.count()

  }
}
