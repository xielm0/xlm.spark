package apps

/**
 * Created by xieliming on 2015/11/11.
 */

import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.{SparkConf, SparkContext}

  object DT {
    def main(args:Array[String]) {
      val sparkConf = new SparkConf()
        .setAppName("DecisionTree")
      val sc = new SparkContext(sparkConf)

      val org_data = sc.textFile("app.db/app_szad_m_dmp_label_childmom_train")   //return array[string]
      val data = org_data.map { line =>
          val parts = line.split("\t").slice(1, 11)
            .map{ t:String=>
              if (t =="NULL" || t=="\\N") 0 else t.toDouble
            }
          LabeledPoint(parts(0), Vectors.dense(parts.tail))
        }

      //统计
      data.map(t=>(t.label,1)).reduceByKey(_+_).collect()
//      0.0,107279628
//      1.0,15198988

      //sampling 类似改变先验概率
      val data0 = data.filter(t=>t.label==0).sample(false,0.1)
      val data1 = data.filter(t=>t.label==1).sample(false,0.3)
      val training=data0.union(data1)

      val numClasses = 2
      val categoricalFeaturesInfo = Map[Int, Int]()
      val impurity = "entropy"  //gini  // variance
      val maxDepth = 5
      val maxBins = 16    //数值分箱数

      //val model_DT1 =DecisionTree.train(training,Classification,impurity,maxDepth,numClasses, sort,maxBins, categoricalFeaturesInfo)
      val model_DT = DecisionTree.trainClassifier(training, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

      //模型评估
      val predict_DT = training.map { point =>
        val prediction = model_DT.predict(point.features)
        (prediction, point.label)
      }

//    要求顺序必须是（预测值, 实际值）
      val matrics_DT=new MulticlassMetrics(predict_DT)
      println(matrics_DT.confusionMatrix)
//      9939303.0  786849.0
//      2555005.0  2004121.0

      //precision  ,recall
      println( matrics_DT.precision(1) , matrics_DT.recall(1) )
//      0.7180732863484738  0.4395844729889018

      val roc_metrics=new BinaryClassificationMetrics(predict_DT)
      println("area under PR:"+roc_metrics.areaUnderPR() +"  AUC:"+roc_metrics.areaUnderROC())
//      area under PR:0.662406195043652  AUC:0.6831132392175151

      //打印决策树
      model_DT.toDebugString

      //save model
      model_DT.save(sc,"app.db/app_szad_m_dmp_label_childmom_model")

      //对所有数据进行预测。
      val predict_new = data.map { point =>
        val prediction = model_DT.predict(point.features)
        ( prediction ,point.label )
      }

      //模型评估
      val matrics_new=new MulticlassMetrics(predict_new)
      println(matrics_new.confusionMatrix)
//      9.9402466E7  7877162.0
//      8515500.0    6683488.0

      //precision  ,recall
      println( matrics_new.precision(1) , matrics_new.recall(1) )
//      0.4590103 0.4397324

      //loss
      matrics_DT.confusionMatrix(1,0)


    }
  }


