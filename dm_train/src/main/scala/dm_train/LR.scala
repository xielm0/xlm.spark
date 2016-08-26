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

    val org_data = sc.textFile("app.db/app_szad_m_dmp_label_childmom_train")   //return array[string]

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
      val flag1 = if( parts(6).toDouble + parts(7).toDouble >0 ) 1.0 else 0.0
      val flag2 = if( parts(9).toDouble >0 ) 1.0 else 0.0

      LabeledPoint(tag, Vectors.dense(gender0,gender1,gender2,age1,age2,age3,age4,age5,age0,user_lev56,user_lev61,user_lev62,user_lev105,
        parts(6).toDouble,parts(7).toDouble,parts(8).toDouble,flag1,flag2))
      }

    //统计
    data.map(t=>(t.label,1)).reduceByKey(_+_).collect()
    //      0.0,107279628
    //      1.0,15198988

    //sampling 类似改变先验概率,样本比例控制在2:1
    val data0 = data.filter(t=>t.label==0).sample(false,0.1)
    val data1 = data.filter(t=>t.label==1).sample(false,0.3)
    val training=data0.union(data1)

    //feature more ,take time more long
    val model_LR = new LogisticRegressionWithLBFGS().setNumClasses(2) .run(training)

    //模型评估
    val predict_LR = training.map(p => (model_LR.predict(p.features), p.label))
    val matrics_LR=new MulticlassMetrics(predict_LR)
    println(matrics_LR.confusionMatrix)
//    1.0443432E7  285772.0
//    3344234.0    1216691.0
//    improve
//    9920763.0  807484.0
//    2603125.0  1958890.0

    //precision  ,recall
    //println( matrics_LR.precision(1) , matrics_LR.recall(1) )
    println("precision(1) = " + matrics_LR.confusionMatrix(1,1)/(matrics_LR.confusionMatrix(1,1)+matrics_LR.confusionMatrix(0,1)) )
    println("recall(1) = " + matrics_LR.confusionMatrix(1,1)/(matrics_LR.confusionMatrix(1,1)+matrics_LR.confusionMatrix(1,0)) )
//    precision(1) = 0.8097976455992594
//    recall(1) = 0.26676408842504534

//    0.7081074359432239
//    0.4293913983185062

    //ROC & AUC
    val roc_metrics=new BinaryClassificationMetrics(predict_LR)
    println("area under PR:"+roc_metrics.areaUnderPR() +"  AUC:"+roc_metrics.areaUnderROC())
//    area under PR:0.6476401144063373  AUC:0.6200645604551069
//    improve
//    area under PR:0.6538730402577937  AUC:0.6770621510129436


    val predict_new = data.map(p => (model_LR.predict(p.features), p.label))

    //模型评估
    val matrics2=new MulticlassMetrics(predict_new)
    println(matrics2.confusionMatrix)
//    1.04423769E8  2855859.0
//    1.1146342E7   4052646.0
//    imporove
//    9.9219225E7  8060403.0
//    8673220.0    6525768.0
    //precision  ,recall
    println("precision(1) = " + matrics2.confusionMatrix(1,1)/(matrics2.confusionMatrix(1,1)+matrics2.confusionMatrix(0,1)) )
    println("recall(1) = " + matrics2.confusionMatrix(1,1)/(matrics2.confusionMatrix(1,1)+matrics2.confusionMatrix(1,0)) )
//    precision(1) = 0.44739417904808604
//    recall(1) = 0.4293554281377155

  }
}
