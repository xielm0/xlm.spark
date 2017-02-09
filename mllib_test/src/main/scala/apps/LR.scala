package apps

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

      val flag1 = if( parts(6).toDouble + parts(7).toDouble >0 ) 1.0 else 0.0 //pv_nums + search_nums
      val flag2 = if( parts(9).toDouble >0 ) 1.0 else 0.0 //gen_fee

      LabeledPoint(tag, Vectors.dense(gender0,gender1,gender2,age1,age2,age3,age4,age5,age0,user_lev56,user_lev61,user_lev62,user_lev105,
        parts(6).toDouble,parts(7).toDouble,parts(8).toDouble,flag1,flag2))
      }


    //sampling 类似改变先验概率,样本比例控制在2:1
    val data0 = data.filter(t=>t.label==0).sample(false,0.1)
    val data1 = data.filter(t=>t.label==1).sample(false,0.3)
    val training=data0.union(data1)

    //feature more ,take time more long
    val model = new LogisticRegressionWithLBFGS
    val model_LR = new LogisticRegressionWithLBFGS().setNumClasses(2) .run(training)


    //模型评估
    val PredAndslabel = data.map(p => (model_LR.predict(p.features), p.label))
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
