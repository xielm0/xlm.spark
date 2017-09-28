package LTR

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{GBTRegressionModel, LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql._

/**
 * Created by xieliming on 2017/9/15.
 */
object lir {

  def pip_fit(df:DataFrame  ,n:Int ):PipelineModel={

    val VectorInputCols2 = Array("gdt_predication","cate_id_ind_vec") ++:input.flag_columns  ++:input.id_columns_vec  ++: input.cross_column_tuple_vec
    val assembler2 = new VectorAssembler().setInputCols(VectorInputCols2).setOutputCol("features2")

    val pipeline2 = new Pipeline().setStages( Array(assembler2))
    val pModel = pipeline2.fit(df)
    // save
    val fs = FileSystem.get(new Configuration())
    fs.delete(new Path( s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/pip_lir" ), true)
    pModel.save(s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/pip_lir")

    return pModel


  }


  def pip_data(df:DataFrame ,n:Int ):DataFrame={
    // load
    val pmodel = PipelineModel.load(s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/pip_lir")
    val transform_df = pmodel.transform(df)

    return transform_df

  }

  def train(trainData:DataFrame,testData:DataFrame, n:Int)={
    val spark =trainData.sparkSession
//    val pModel1=input.pip_fit(trainData,n)
//    val new_train = input.genearate_new_column(trainData,"apply",n)
//    val new_test = input.genearate_new_column(testData,"apply",n)
//    val transform_train=pModel1.transform(new_train )   // fit
//    val transform_test=pModel1.transform(new_test )
    val transform_train=input.pip_data(trainData,n)
    val transform_test=input.pip_data(testData,n)
    val (transform_train1,transform_test1)=gbdt.train(transform_train,transform_test,n)
    //
    val pModel2= pip_fit(transform_train1,n)
    val transform_train2= pModel2.transform(transform_train1 ).select("label","features2")
    val transform_test2 = pModel2.transform(transform_test1 ).select("label","features2")

    val lr = new LinearRegression()
      .setMaxIter(30)
      .setRegParam(1e-5)
      .setElasticNetParam(0)
      .setFeaturesCol("features2")
      .setLabelCol("label")
      .setStandardization(false)

    //      val paramGrid = new  ParamGridBuilder()
    //        .addGrid(lr.regParam, Array(0.1,0.01,0.003))
    //        .addGrid(lr.elasticNetParam,Array(0.0, 0.5, 1.0))
    //        .build()
    //
    //      val tvs = new TrainValidationSplit( )
    //        .setEstimator( lr )
    //        .setEstimatorParamMaps( paramGrid )
    //        .setEvaluator( new BinaryClassificationEvaluator().setLabelCol("label").setMetricName("areaUnderROC"))
    //        .setTrainRatio( 0.75 )
    //
    //      val cvModel = tvs.fit(transform_train2)
    //      val lrModel_best = cvModel.bestModel.asInstanceOf[LogisticRegressionModel]
    //      println("best RegParam ="+lrModel_best.getRegParam)
    //      println("best ElasticNetParam ="+lrModel_best.getElasticNetParam)
    //      println("best Intercept ="+lrModel_best.getFitIntercept)
    //
    //      val lrModel=lrModel_best

    val lrModel=lr.fit(transform_train2)
    // println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}") // VectorInputCols(0)

    // loss
    val trainingSummary = lrModel.summary
    println("loss:\n")
    trainingSummary.objectiveHistory.foreach(loss => println(loss))

    val predictions = lrModel.transform(transform_test2)

    val scoreAndLabels2 = predictions.select( "prediction","label").rdd.map{
      t=>(t.getAs[Double]("prediction"),t.getAs[Double]("label"))
    }
    val auc2 = new BinaryClassificationMetrics(scoreAndLabels2).areaUnderROC
    println("Area under ROC:gbdt+LR = " + auc2)

    // save
    val fs = FileSystem.get(new Configuration())
    fs.delete(new Path( s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/lir" ), true)
    lrModel.save( s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/lir")

  }

  /**
   *
   * @param df
   * @param n 代表哪个模型
   */
  def predict_res(df:DataFrame, n:Int):DataFrame={
    val spark = df.sparkSession
    val transform_apply = input.pip_data(df,n)
    //load model
    val gbtModel  = GBTRegressionModel.load(s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/gbdt")
    val transform_apply1 = gbtModel.transform(transform_apply).withColumnRenamed("prediction","gdt_predication")
    // load
    val transform_apply2 =pip_data(transform_apply1 ,n)
    val lrModel = LinearRegressionModel.load( s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/lir")
    val predictions =lrModel.transform(transform_apply2).select("user_id","sku_Id", "prediction")
    predictions
  }


  def main(args: Array[String]) {
    val spark = SparkSession.builder()
      .appName("CPM.LTR.lir")
      .config("spark.rpc.askTimeout", "500")
      .config("spark.speculation", "false")
      .config("spark.memory.fraction", "0.8")
      .config("spark.memory.storageFraction", "0.5")
      .enableHiveSupport()
      .getOrCreate()

    val model_stage: String = args(0)
    val n:Int =args(1).toInt

    if (model_stage=="train"){
      val (trainData,testData)=input.get_train_input(spark,n)
      train(trainData,testData,n)
    }

    else {
      val apply_data = input.get_apply_input(spark,n)
      //apply_data.persist(StorageLevel.MEMORY_AND_DISK)
      val predictions =predict_res(apply_data,n)

      val model_id:String = args(2)
      // top10
      val k=10
      predictions.createOrReplaceTempView("tmp_table")
      spark.sql("use app")
      spark.sql("set mapreduce.output.fileoutputformat.compress=true")
      spark.sql("set hive.exec.compress.output=true")
      spark.sql("set mapred.output.compression.codec=com.hadoop.compression.lzo.LzopCodec")
      val ss =
        s"""
           |insert overwrite table app.app_szad_m_dyrec_rank_predict_res partition(model_id=${model_id},user_type=1)
           |select user_id,sku_Id,prediction,rn
           |from(select user_id,sku_Id,prediction,
           |            row_number() over (partition by user_id order by prediction desc )  rn
           |      from tmp_table
           |    )t
           |where rn <= ${k}
        """.stripMargin
      spark.sql(ss)
    }


  }





}
