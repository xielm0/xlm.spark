package dm_train

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by xieliming on 2015/11/13.
 */
object LR {
  def main(args:Array[String]) {
    val sparkConf = new SparkConf().setAppName("LR")
    val sc = new SparkContext(sparkConf)

    val data = sc.textFile("app.db/app_szad_m_dmp_label_gender_xlm") //return array[string]
    //读取样本数据
    val parseData = data.map { line =>
        val parts = line.split('\t').map(_.toDouble)
        LabeledPoint(parts(0), Vectors.dense(parts.tail))
      }

    val splits = parseData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training=splits(0).cache()

    val model_lr = new LogisticRegressionWithLBFGS().setNumClasses(3) .run(training)

    //模型评估
    val predict1_lr = training.map(p => (model_lr.predict(p.features), p.label))
    val matrics=new MulticlassMetrics(predict1_lr)
    println(matrics.confusionMatrix)
//    1390040.0  103444.0  362138.0
//    164504.0   254439.0  323998.0
//    686105.0   165530.0  1770453.0
    //计算准确率\召回率
    (0 until 3).map(
      sex => (matrics.precision(sex), matrics.recall(sex))
    ).foreach(println)
//    (0.6203738291896678,0.749096529357811)
//    (0.4861151710026308,0.34247537826018487)
//    (0.7206956475014746,0.6752073156964984)

    //ROC & AUC
    val roc_metrics=new BinaryClassificationMetrics(predict1_lr)
    println("area under PR:"+roc_metrics.areaUnderPR() +"  AUC:"+roc_metrics.areaUnderROC())
//    area under PR:0.8705045402454293  AUC:0.7533292275971389

    /*val data2= sc.textFile("app.db/app_szad_f_dmp_gender_xlm_yuce_m/dt=2015-11-01")
    val test = data2.map { line =>
      val parts = line.split('\t').slice(2,17).map(_.toDouble)
      LabeledPoint(parts(0), Vectors.dense(parts.tail))
    }*/

    val test=splits(1)
    val predict2 = test.map(p => (model_lr.predict(p.features), p.label))

    //模型评估
    val matrics2=new MulticlassMetrics(predict2)
    println(matrics2.confusionMatrix)
    //计算准确率\召回率
    (0 until 2).map(
      sex => (matrics2.precision(sex), matrics.recall(sex))
    ).foreach(println)

    val accuracy = 1.0 * predict2.filter(x => x._1 == x._2).count()/ test.count()

  }
}
