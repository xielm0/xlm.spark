package apps

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
    val sparkConf = new SparkConf().setAppName("byes")
    val sc = new SparkContext(sparkConf)

    val org_data = sc.textFile("app.db/app_szad_m_dmp_label_childmom_train")   //return array[string]


    //bayes，若是二分类，则要求输入分类型特征，而且值必须是0/1 ，
    val data = org_data.map { line =>
      val parts = line.split("\t")
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
      //分箱
      val pv_nums1 = if (parts(6).toDouble > 0) 1 else 0
      val pv_nums2 = if (parts(6).toDouble > 6) 1 else 0
      val pv_nums3 = if (parts(6).toDouble > 26) 1 else 0
      val gen_fee1 = if (parts(9).toDouble > 0 ) 1 else 0
      val gen_fee2 = if (parts(9).toDouble > 8900 ) 1 else 0
      val gen_fee3 = if (parts(9).toDouble > 19900 ) 1 else 0

      LabeledPoint(tag, Vectors.dense(gender0,gender1,gender2,age1,age2,age3,age4,age5,age0,user_lev56,user_lev61,user_lev62,user_lev105,
        pv_nums1,pv_nums2,pv_nums3,gen_fee1,gen_fee2,gen_fee3))
    }

    //统计
    data.map(t=>(t.label,1)).reduceByKey(_+_).collect()
    //      0.0,107279628
    //      1.0,15198988
    //sampling 类似改变先验概率
    val data0 = data.filter(t=>t.label==0).sample(false,0.1)
    val data1 = data.filter(t=>t.label==1).sample(false,0.3)
    val training=data0.union(data1)

    //默认是modelType = "multinomial" 多项式分布
    val model_byes = NaiveBayes.train(training, lambda = 1.0, modelType = "bernoulli")

    //模型评估
    val predict_bayes = training.map(p => (model_byes.predict(p.features),p.label))
    val matrics_bayes=new MulticlassMetrics(predict_bayes)
    println(matrics_bayes.confusionMatrix)
//    9548807.0  1179667.0
//    2289416.0  2270719.0

    //precision  ,recall
    println("precision(1) = " + matrics_bayes.confusionMatrix(1,1)/(matrics_bayes.confusionMatrix(1,1)+matrics_bayes.confusionMatrix(0,1)) )
    println("recall(1) = " + matrics_bayes.confusionMatrix(1,1)/(matrics_bayes.confusionMatrix(1,1)+matrics_bayes.confusionMatrix(1,0)) )
//    precision(1) = 0.6581057887436362
//    recall(1) = 0.49794995104311607

    //ROC & AUC
    val roc_metrics=new BinaryClassificationMetrics(predict_bayes)
    println("area under PR:"+roc_metrics.areaUnderPR() +"  AUC:"+roc_metrics.areaUnderROC())
//    area under PR:0.6530121552282322  AUC:0.6939809766338583

    val predict_new = data.map(p => (model_byes.predict(p.features), p.label))

    //模型评估
    val matrics2=new MulticlassMetrics(predict_new)
    println(matrics2.confusionMatrix)
//    9.5478709E7  1.1800919E7
//    7627139.0    7571849.0

    //precision  ,recall
    println("precision(1) = " + matrics2.confusionMatrix(1,1)/(matrics2.confusionMatrix(1,1)+matrics2.confusionMatrix(0,1)) )
    println("recall(1) = " + matrics2.confusionMatrix(1,1)/(matrics2.confusionMatrix(1,1)+matrics2.confusionMatrix(1,0)) )
//    precision(1) = 0.390850135613042
//    recall(1) = 0.49818112890147687
  }
}
