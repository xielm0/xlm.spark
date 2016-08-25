package dm_train

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
      val flag1 = if( parts(6).toDouble + parts(7).toDouble >0 ) 1.0 else 0.0
      val flag2 = if( parts(9).toDouble >0 ) 1.0 else 0.0

      LabeledPoint(tag, Vectors.dense(gender0,gender1,gender2,age1,age2,age3,age4,age5,age0,user_lev56,user_lev61,user_lev62,user_lev105,
        parts(6).toDouble,parts(7).toDouble,parts(8).toDouble,flag1,flag2))
    }

    //统计
    data.map(t=>(t.label,1)).reduceByKey(_+_).collect()
    //      0.0,107279628
    //      1.0,15198988
    //sampling 类似改变先验概率
    val data0 = data.filter(t=>t.label==0).sample(false,0.1)
    val data1 = data.filter(t=>t.label==1).sample(false,0.3)
    val training=data0.union(data1)

    //默认是modelType = "multinomial"
    val model_byes = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")
    //multinomial 多项式分布

    //模型评估
    val predict_bayes = training.map(p => (model_byes.predict(p.features),p.label))
    val matrics_bayes=new MulticlassMetrics(predict_bayes)
    println(matrics_bayes.confusionMatrix)

    //precision  ,recall
    println( matrics_bayes.precision(1) , matrics_bayes.recall(1) )

    //ROC & AUC
    val roc_metrics=new BinaryClassificationMetrics(predict_bayes)
    println("area under PR:"+roc_metrics.areaUnderPR() +"  AUC:"+roc_metrics.areaUnderROC())
//    area under PR:0.8109519106138725  AUC:0.6678262266955499

    /*  val data2= sc.textFile("app.db/app_szad_f_dmp_gender_xlm_yuce_m/dt=2015-11-01")
    val test = data2.map { line =>
      val parts = line.split('\t').slice(2,17).map(_.toDouble)
      LabeledPoint(parts(0), Vectors.dense(parts.tail))
    }*/
    val splits = data.randomSplit(Array(0.7, 0.3))
    val test=splits(1)
    val predict2 = test.map(p => (model_byes.predict(p.features), p.label))

    //模型评估
    val matrics2=new MulticlassMetrics(predict2)
    println(matrics2.confusionMatrix)

    //precision  ,recall
    println( matrics2.precision(1) , matrics2.recall(1) )

    val accuracy = 1.0 * predict2.filter(x => x._1 == x._2).count()/ test.count()

  }
}
