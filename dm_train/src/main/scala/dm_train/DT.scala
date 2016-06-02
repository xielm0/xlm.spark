package dm_train

/**
 * Created by xieliming on 2015/11/11.
 */

import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.{SparkConf, SparkContext}

  /**
   * Created by wangjing15 on 2015/11/3.
   */

  //man 90w ,woman 90w
  object DT {
    def main(args:Array[String]) {
      val sparkConf = new SparkConf().setAppName("DecisionTree")
      val sc = new SparkContext(sparkConf)

      val data = sc.textFile("app.db/app_szad_m_dmp_label_gender_xlm")   //return array[string]
      val parseData = data.map { line =>
          val parts = line.split('\t').slice(1, 15).map(_.toDouble)
          LabeledPoint(parts(0), Vectors.dense(parts.tail))
        }
      //男：1855622  女 742941 中性 2622088
      //注册：2545307  女694243 中性 1981101
      // Split the data into training a
      val splits = parseData.randomSplit(Array(0.6, 0.4))
      val training=splits(0).cache()


      //return org.apache.spark.mllib.regression.LabeledPoint = (1.0,[0.0,0.0,0.0,0.0,0.0,8000.0,0.0,0.0,0.0,0.0,0.0,854.0])
      val numClasses = 3
      val categoricalFeaturesInfo = Map[Int, Int]()
      val impurity = "entropy"  //gini
      val maxDepth = 10
      val maxBins = 200    //数值分箱数

      //val model_DT =DecisionTree.train(training,Algo.Classification, Entropy ,maxDepth)
      val model_DT = DecisionTree.trainClassifier(training, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)
      //模型评估
      val predict1_DT = training.map { point =>
        val prediction = model_DT.predict(point.features)
        (  prediction ,point.label )
      }
      val matrics_DT=new MulticlassMetrics(predict1_DT)
      println(matrics_DT.confusionMatrix)
//      1466849.0  65360.0   323413.0
//      178485.0   404706.0  159750.0
//      714854.0   171113.0  1736121.0


      //计算准确率\召回率
      //要求顺序必须是（预测值, 实际值）
      (0 until 3).map(
        sex => (matrics_DT.precision(sex), matrics_DT.recall(sex))
      ).foreach(println)

//      (0.6214966773833271,0.790489119012385)
//      (0.6311903540197044,0.5447350462553554)
//      (0.7822887922411011,0.662113933628467)

      //ROC & AUC
      //要求顺序必须是（预测值, 实际值）
      val roc_metrics=new BinaryClassificationMetrics(predict1_DT)
      println("area under PR:"+roc_metrics.areaUnderPR() +"  AUC:"+roc_metrics.areaUnderROC())
//      area under PR:0.8696296208831779  AUC:0.7575162110601223

      //运用模型去预测用户性别
      val data2= sc.textFile("app.db/app_szad_m_dmp_label_gender_xlm/dt=2015-11-01")
      val test = data2.map { line =>
        val parts = line.split('\t').slice(1,56).map(_.toDouble)
        LabeledPoint(parts(0), Vectors.dense(parts.tail))
      }
//      val test=splits(1)
      val predict2 = test.map { point =>
        val prediction = model_DT.predict(point.features)
        ( prediction ,point.label )
      }

      //模型评估
      val matrics2=new MulticlassMetrics(predict2)
      println(matrics2.confusionMatrix)
//      2498017.0  578434.0  0.0
//      942803.0   901484.0  0.0
//      1515854.0  684488.0  0.0

      //计算准确率\召回率
      (0 until 3).map(
        sex => (matrics2.precision(sex), matrics2.recall(sex))
      ).foreach(println)
//      (0.5039704043477542,0.81198010304731)
//      (0.4165041124447077,0.4887981100555391)
//      (0.0,0.0)

      val testErr = predict2.filter(r => r._1 != r._2).count().toDouble / test.count
      println("test Error = " + testErr)


      // Save and load model
      // model_DT.save(sc, "app.db/app_szad_f_dmp_gender_xlm_yuce_m/")
      //load
      //val sameModel = DecisionTreeModel.load(sc, "app.db/app_szad_f_dmp_gender_forecast_v1_4/dt=2015-10-20/")

    }
  }


