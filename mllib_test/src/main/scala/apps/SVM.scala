package apps

import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by xieliming on 2016/12/20.
 */
object SVM {
  def main(args:Array[String]) {
    val sparkConf = new SparkConf().setAppName("LR")
    val sc = new SparkContext(sparkConf)

    val org_data = sc.textFile("app.db/app_szad_m_dmp_label_childmom_train").map(_.split("\t"))   //return array[string]

    val data = org_data.map { parts =>
      val tag = parts(1).toDouble
      val gender0 = if(parts(2)=="0") 1 else 0
      val gender1 = if(parts(2) =="1") 1 else 0
      val gender2 = if(parts(2) !="0" & parts(2) !="1") 1 else 0
      val age1 = if(parts(3).toDouble >= 10 & parts(3).toDouble < 20) 1 else 0
      val age2 = if(parts(3).toDouble >= 20 & parts(3).toDouble < 30) 1 else 0
      val age3 = if(parts(3).toDouble >= 30 & parts(3).toDouble < 40) 1 else 0
      val age4 = if(parts(3).toDouble >= 40 & parts(3).toDouble < 50) 1 else 0
      val age5 = if(parts(3).toDouble >= 50  ) 1 else 0
      val age0 = if(parts(3).toDouble <10 ) 1 else 0
      val user_lev56 = if (parts(4)=="56") 1 else 0
      val user_lev61 = if (parts(4)=="61") 1 else 0
      val user_lev62 = if (parts(4)=="62") 1 else 0
      val user_lev105 = if (parts(4)=="105") 1 else 0
      val pv_nums = parts(6).toDouble
      val search_nums = parts(6).toDouble
      val gen_nums = parts(6).toDouble
      val gen_fee = parts(6).toDouble
      val favorite_tag = parts(6).toDouble

      LabeledPoint(tag, Vectors.dense(gender0,gender1,gender2,age1,age2,age3,age4,age5,age0,user_lev56,user_lev61,user_lev62,user_lev105,
        pv_nums,search_nums,gen_nums,gen_fee,favorite_tag  ))
    }

    //split data
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    //spark 只支持linear svm
    //svm 要做距离计算，计算间隔
//  val model = SVMWithSGD.train(input, numIterations, 1.0, 0.01, 1.0)
    // regParm = 1/2C
    val lambda = 0.01
    val model = SVMWithSGD.train(training, numIterations=100,stepSize=0.1,regParam=lambda,miniBatchFraction=0.5)

    //模型评估
    val PredAndslabel = data.map(p => (model.predict(p.features), p.label))
    val m_matrics=new MulticlassMetrics(PredAndslabel)
    println(m_matrics.confusionMatrix)
    //    9.9221008E7  8058620.0
    //    8674722.0    6524266.0
    println( "precision=" +m_matrics.precision  )
    //    precision=0.8633774405158203
    println( m_matrics.precision(1) , m_matrics.recall(1) )
    //    (0.4473919634289125,0.4292566057687525)

    //ROC & AUC
    val b_metrics=new BinaryClassificationMetrics(PredAndslabel)
    // AUROC
    val auROC = b_metrics.areaUnderROC
    println("Area under ROC = " + auROC)
    //    auROC: Double = 0.6770693546001783

  }

}
