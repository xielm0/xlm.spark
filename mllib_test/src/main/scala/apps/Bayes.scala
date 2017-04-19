package apps

import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}


/**
 * Created by xieliming on 2015/11/12.
 */
object Bayes {
  def main(args:Array[String]) {
    val sparkConf = new SparkConf().setAppName("byes")
    val sc = new SparkContext(sparkConf)

    val org_data = sc.textFile("app.db/app_szad_m_dmp_label_childmom_train").map(_.split("\t"))   //return array[string]

    //bayes，若是二分类，则要求输入分类型特征，而且值必须是0/1 ，
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

    //sampling 类似改变先验概率
    val data0 = data.filter(t=>t.label==0).sample(false,0.1)
    val data1 = data.filter(t=>t.label==1).sample(false,0.3)
    val training=data0.union(data1)

    //默认是modelType = "multinomial" 多项式分布
    val model_byes = NaiveBayes.train(training, lambda = 1.0, modelType = "bernoulli")

    //模型评估
    val PredAndslabel = data.map(p => (model_byes.predict(p.features), p.label))

    //    要求顺序必须是（预测值, 实际值）
    val m_matrics=new MulticlassMetrics(PredAndslabel)
    println(m_matrics.confusionMatrix)
//    9.5478709E7  1.1800919E7
//    7627139.0    7571849.0
    println( "precision=" +m_matrics.precision  )
//    precision=0.8413759182255945
    println( m_matrics.precision(1) , m_matrics.recall(1) )
//    (0.390850135613042,0.49818112890147687)

    val b_metrics=new BinaryClassificationMetrics(PredAndslabel)
    // AUROC
    val auROC = b_metrics.areaUnderROC
    println("Area under ROC = " + auROC)
//    Area under ROC = 0.6940898191088548



  }
}
