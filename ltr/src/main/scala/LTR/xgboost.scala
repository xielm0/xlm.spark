package LTR

import ml.dmlc.xgboost4j.scala.spark.{XGBoost, XGBoostModel}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.collection.mutable.ArrayBuffer


object xgboost {

  val num_round=10
  def train(transform_train:DataFrame,transform_test:DataFrame,n:Int):(XGBoostModel,DataFrame,DataFrame)={
    val spark=transform_train.sparkSession

    //xgboost
    val paramMap = List(
      "eta" -> 0.1f
      ,"max_depth" -> 6
      ,"objective" -> "binary:logistic"
      //,"objective" -> "rank:pairwise"
    ).toMap

    val xgbModel = XGBoost.trainWithDataFrame( transform_train, paramMap, num_round, nWorkers=10 ,useExternalMemory = true )
      .setFeaturesCol("features").setLabelCol("label")

    //save
    val fs = FileSystem.get(new Configuration())
    fs.delete(new Path( s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/_xgb" ), true)
    xgbModel.saveModelAsHadoopFile(s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/_xgb")(spark.sparkContext)


    val transform_train1 = xgbModel.transform(transform_train )
    val transform_test1 = xgbModel.transform(transform_test )
    //auc
    val scoreAndLabels = transform_test1.select( "probabilities","label").rdd.map{
      case Row(score: Vector, label: Double) => (score(1), label)
    }
    //    val scoreAndLabels = xgb_predictions.select( "prediction","label").rdd.map{
    //      t=>(t.getAs[Float]("prediction").toDouble,t.getAs[Double]("label"))
    //    }
    val BinaryMetrics=new BinaryClassificationMetrics(scoreAndLabels)
    val auc = BinaryMetrics.areaUnderROC
    println("Area under ROC = " + auc)

    return(xgbModel,transform_train1,transform_test1)

  }


  def predict_res(transform_apply:DataFrame,n:Int):DataFrame={
    val spark =transform_apply.sparkSession

    // load model
    val xgbModel  = XGBoost.loadModelFromHadoopFile(s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/_xgb")(spark.sparkContext)
    //
    val predictions = xgbModel.transform(transform_apply )
    predictions

  }



  val leafColumnsBuffer =ArrayBuffer[String]()
  for (i <- Range(0,num_round,1)) {
    leafColumnsBuffer += s"tree_${i}"
  }
  val leafColumns =leafColumnsBuffer.toArray
  val leafColumns_ind =  leafColumns.map( _ + "_ind")
  val leafColumns_vec =  leafColumns_ind.map( _ + "_vec")

  /**
   *将xgbosst生成的 predLeaf 列（类型是向量） 转换成  num_round 个列
   * @param transform_df : 已经经过pip_data处理好的dataframe
   */
  def create_leaf(xgbModel:XGBoostModel,transform_df:DataFrame):DataFrame={
    //
    // create features , col name = predLeaf ,是一个Seq[Float],记录了在每棵树的位置。
    // [3,5,7,...,10]
    // transformLeaf this function 特别耗内存。
    val transform_Leaf = xgbModel.transformLeaf(transform_df)

    //
    var df_a:DataFrame=transform_Leaf
    var df_b:DataFrame=transform_Leaf
    // 将 predLeaf 变成 num_round 个字段
    for (i <- Range(0,num_round,1)) {
      df_b  = df_a.withColumn(
        s"tree_${i}",
        // 只要第i个位置的数字
        udf( (predLeaf: Seq[Float])  => predLeaf(i).toDouble)
          .apply(col("predLeaf"))
      )
      df_a=df_b
    }

    val xgb_leaf= df_a
    xgb_leaf
  }

  def main(args: Array[String]) {
    val spark = SparkSession.builder()
      .appName("cpm.ltr.xgboost")
      .config("spark.rpc.askTimeout", "500")
      .config("spark.speculation", "false")
      .config("spark.memory.fraction", "0.75")
      .config("spark.memory.storageFraction", "0.5")
      .enableHiveSupport()
      .getOrCreate()

    val model_stage: String = args(0)
    val n :Int = args(1).toInt
    if (model_stage=="train"){
      val (trainData,testData)=input.get_train_input(spark,n)

      val pModel1=input.pip_fit(trainData,n)
      val new_train = input.genearate_new_column(trainData,"apply",n)
      val new_test = input.genearate_new_column(testData,"apply",n)
      val transform_train=pModel1.transform(new_train)
      val transform_test=pModel1.transform(new_test)

      train(transform_train,transform_test,n)


    }else {
      val apply_data=input.get_apply_input(spark,n)
      val transform_apply=input.pip_data(apply_data,n)
      val predictions=predict_res(transform_apply,n)
      predictions.select("user_id","sku_id","probabilities").show(10)
    }
  }
}
