package test

/**
 * Created by xieliming on 2016/6/1.
 */
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by xieliming on 2015/12/30.
 * 1先用分类模型，DT,OR byese
 */
object classific_child {
  /*param1 :train, apply
  * */
  def classific_train= {
    val sparkConf = new SparkConf().setAppName("Childmom_tag")
    val sc = new SparkContext(sparkConf)

    //dt
    //读取数据
    //    val org_data = sc.textFile("app.db/app_szad_m_dmp_label_childmom_train_dt")
    //    val org_data_true=org_data.filter(line=>line.split("\t")(2)=="1")
    //    val org_data_false=org_data.filter{line=>
    //      val parts=line.split("\t")
    //      parts(2)=="0" && parts(1)=="0"}.randomSplit(Array(0.1,0.9))
    //
    //    val org_data_sample=org_data_true.union(org_data_false(0))
    //    //数据处理
    //    val dt_train_data = org_data_sample.map {
    //      line =>
    //        val parts = line.split("\t").slice(1, 17).map(_.toDouble)
    //        LabeledPoint(parts(1), Vectors.dense(parts.slice(2, 16)))
    //    }
    //    //统计数据分布
    ////    dt_train_data.map(t => (t.label, 1)).reduceByKey(_ + _).collect
    //
    //    val numClasses = 2
    //    val categoricalFeaturesInfo = Map[Int, Int]() //损失矩阵
    //    val impurity = "entropy" //gini entropy
    //    val maxDepth = 6
    //    val maxBins = 200 //数值分箱数
    //
    //    //val model_DT =DecisionTree.train(training,Algo.Classification, Entropy ,maxDepth)
    //    val model_DT = DecisionTree.trainClassifier(dt_train_data, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)
    //    //打印树结构
    //    println(model_DT.toDebugString)
    //    //模型评估
    //    val predict_fact_dt1 = dt_train_data.map ( p =>(model_DT.predict(p.features), p.label) )
    //
    //    val matrics_DT = new MulticlassMetrics(predict_fact_dt1) //MulticlassMetrics(predictionAndLabels)
    //    println("DT confusionMatrix")
    //    println(matrics_DT.confusionMatrix) //confusion 混淆
    //    //    行:实际 ，列:预测
    ////    7327303.0  243367.0
    ////    747165.0   595265.0
    //
    //    //计算准确率\召回率
    //    //要求顺序必须是（预测值, 实际值）
    //    (0 until 2).map(
    //      t => (matrics_DT.precision(t), matrics_DT.recall(t))
    //    ).foreach(println)
    //
    ////      (0.8610650199201635,0.9346274795347335)
    ////      (0.757426636214515,0.5751122963580969)
    //
    //    //ROC & AUC
    //    //要求顺序必须是（预测值, 实际值）
    //    val roc_metrics = new BinaryClassificationMetrics(predict_fact_dt1)
    //    println("area under PR:" + roc_metrics.areaUnderPR() + "  AUC:" + roc_metrics.areaUnderROC())
    //    //  area under PR:0.6185280036118356  AUC:0.7056387306141403
    //
    //删除mode文件
    val conf = new Configuration()
    val fs = FileSystem.get(conf)
    //    fs.delete(new Path( "app.db/app_szad_m_dmp_label_childmom_model/model=DT/data"),true)
    //    fs.delete(new Path( "app.db/app_szad_m_dmp_label_childmom_model/model=DT/metadata"),true)
    //
    //    //模型保存
    //    model_DT.save(sc,"hdfs://ns3/user/jd_ad/app.db/app_szad_m_dmp_label_childmom_model/model=DT")

    //      ****************************************************************
    //byese
    //类目表
    val vocabRDD = sc.textFile("hdfs://ns3/user/jd_ad/app.db/app_szad_m_dmp_label_childmom_cate")
    val vocabSize = vocabRDD.count().toInt
    val vocab = vocabRDD.zipWithIndex().collect.toMap
    //读取数据
    val org_data = sc.textFile("app.db/app_szad_m_dmp_label_childmom_train_byes")
    val byes_train_data = org_data.map {
      line =>
        // val line=org_data.first
        val parts = line.split("\t")
        val cates = parts(3).split(",").map {
          cateCnt =>
            //val cateCnt=parts(17).split(",")(0)
            val s = cateCnt.split(":")
            (vocab(s(0)).toInt, s(1).toDouble)
        }
        LabeledPoint(parts(1).toInt, Vectors.sparse(vocabSize, cates))
    }

    val model_byes = NaiveBayes.train(byes_train_data, lambda = 1.0 /*, modelType = "bernoulli"*/)

    //模型评估
    val predict_fact_byes1 = byes_train_data.map ( p =>(model_byes.predict(p.features), p.label) )

    val matrics_byes1 = new MulticlassMetrics(predict_fact_byes1) //MulticlassMetrics(predictionAndLabels)
    println("byes confusionMatrix")
    println(matrics_byes1.confusionMatrix)
    //    9.1030033E7  2.262931E7
    //    3497205.0    2947530.0

    (0 until 2).map(
      t => (matrics_byes1.precision(t), matrics_byes1.recall(t))
    ).foreach(println)
    //    (0.9911462485366245,0.8049864502998988)
    //    (0.051303361005121194,0.5945799780994168)
    //    (0.9630032033729791,0.8009023332116216)
    //    (0.1152421487564531,0.4573547244378551)


    //ROC & AUC
    //要求顺序必须是（预测值, 实际值）
    val roc_metrics_b = new BinaryClassificationMetrics(predict_fact_byes1)
    println("area under PR:" + roc_metrics_b.areaUnderPR() + "  AUC:" + roc_metrics_b.areaUnderROC())
    //    area under PR:0.32647441805979166  AUC:0.6997832141996577

    //删除mode文件
    fs.delete(new Path( "app.db/app_szad_m_dmp_label_childmom_model/model=byes/data"),true)
    fs.delete(new Path( "app.db/app_szad_m_dmp_label_childmom_model/model=byes/metadata"),true)

    //模型保存
    model_byes.save(sc,"hdfs://ns3/user/jd_ad/app.db/app_szad_m_dmp_label_childmom_model/model=byes")

    sc.stop()

  }


  /*预测*/
  def classific_predict = {

    val sparkConf = new SparkConf().setAppName("Childmom_tag")
    val sc = new SparkContext(sparkConf)

    //    //dt
    //    //加载数据
    //    val org_data = sc.textFile("app.db/app_szad_m_dmp_label_childmom_train_dt")
    //    //加载模型
    //    val model_DT=DecisionTreeModel.load(sc,"hdfs://ns3/user/jd_ad/app.db/app_szad_m_dmp_label_childmom_model/model=DT")

    //    //数据处理
    //    val dt_predict_data = org_data.map {
    //      line =>
    //        val content=line.split("\t").slice(0,17)
    //        val user_id=content(0)
    //        val parts = content.slice(1, 17).map(_.toDouble)
    //        (user_id,LabeledPoint(parts(1), Vectors.dense(parts.slice(2, 16))) )
    //    }
    //
    //    val dt_result = dt_predict_data.map { point =>
    //      val prediction = model_DT.predict(point._2.features)
    //      (point._1, prediction )
    //    }

    //      ******************************************
    //byes模型应用
    //加载数据
    val org_data = sc.textFile("app.db/app_szad_m_dmp_label_childmom_apply_byes")
    //加载模型
    val model_byes=NaiveBayesModel.load(sc,"hdfs://ns3/user/jd_ad/app.db/app_szad_m_dmp_label_childmom_model/model=byes")
    //加载字典
    val vocabRDD = sc.textFile("hdfs://ns3/user/jd_ad/app.db/app_szad_m_dmp_label_childmom_cate")
    val vocabSize = vocabRDD.count().toInt
    val vocab = vocabRDD.zipWithIndex().collect.toMap
    //数据处理
    val byese_predict_data = org_data.map {
      line =>
        // val line=org_data.first
        val parts = line.split("\t")
        val user_id=parts(0)
        val cates = parts(3).split(",").map {
          cateCnt =>
            //val cateCnt=parts(17).split(",")(0)
            val s = cateCnt.split(":")
            (vocab(s(0)).toInt, s(1).toDouble)
        }
        (user_id ,LabeledPoint(parts(1).toInt, Vectors.sparse(vocabSize, cates)))
    }

    val byes_result = byese_predict_data.map { point =>
      val prediction = model_byes.predict(point._2.features)
      (point._1,prediction )
    }

    //      ******************************************
    // 用hive将结果插入表
    val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)
    import hiveContext.implicits._


    // 将结果插入表
    hiveContext.sql("use app")
    hiveContext.sql("set mapred.output.compress=true")
    hiveContext.sql("set hive.exec.compress.output=true")
    hiveContext.sql("set mapred.output.compression.codec=com.hadoop.compression.lzo.LzopCodec")

    //    val df=dt_result.toDF("user_id","label").registerTempTable("tmp_table")
    //    hiveContext.sql("INSERT overwrite TABLE app_szad_m_dmp_label_childmom_result partition(model='DT') select user_id,label from tmp_table")


    // 将结果插入表
    val df_byes= byes_result.toDF("user_id","label").registerTempTable("tmp_table2")
    hiveContext.sql("INSERT overwrite TABLE app_szad_m_dmp_label_childmom_result partition(model='byes') select user_id,label from tmp_table2")

    sc.stop()
  }

  /*
   nohup spark-submit --master yarn-client --class childmom_tag.classific_child \
   --num-executors 2 --executor-memory 10g --executor-cores 2 \
   /home/jd_ad/xlm/spark_dmp_label_childmom.jar  train  &

   nohup spark-submit --master yarn-client --class childmom_tag.classific_child  /home/jd_ad/xlm/spark_dmp_label_childmom.jar  predict  &
    */
  def main(args:Array[String])= {

    val run_type=args(0)
    if(run_type=="train") classific_train
    else if(run_type=="predict") classific_predict

  }
}
