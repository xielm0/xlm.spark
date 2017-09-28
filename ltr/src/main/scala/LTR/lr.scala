package LTR


import ml.dmlc.xgboost4j.scala.spark.XGBoost
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

/**
 * Created by xieliming on 2017/9/15.
 */
object lr {

  val sparse_columns = Array("cate_id_ind_vec") ++: input.flag_columns ++: input.id_columns_vec ++: input.cross_column_tuple_vec
  def pip_fit(df:DataFrame,n:Int):PipelineModel={
    println("starting lr's pip_fit")
    df.cache()

    val string_index2: Array[org.apache.spark.ml.PipelineStage] = xgboost.leafColumns.map(
      cname => new StringIndexer()
        .setInputCol(cname)
        .setOutputCol(s"${cname}_ind").setHandleInvalid("skip")
    )

    val Onehot_Encoder2: Array[org.apache.spark.ml.PipelineStage] = xgboost.leafColumns_ind.map(
      cname => new OneHotEncoder()
        .setInputCol(cname)
        .setOutputCol(s"${cname}_vec")
    )
    //  Assembler
    val VectorInputCols2 = xgboost.leafColumns_vec ++: sparse_columns
    val assembler2 = new VectorAssembler().setInputCols(VectorInputCols2).setOutputCol("features2")

    //
    val pipeline2 = new Pipeline().setStages(string_index2 ++: Onehot_Encoder2 ++: Array(assembler2))
    val pModel = pipeline2.fit(df)
    // save
    val fs = FileSystem.get(new Configuration())
    fs.delete(new Path(s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/pip_lr"), true)
    pModel.save(s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/pip_lr")

    df.unpersist()
    return pModel

  }

  def pip_data(df:DataFrame,n:Int):DataFrame={
    println("starting lr's pip_data")
    // load
    val pmodel = PipelineModel.load(s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/pip_lr")
    val transform_df = pmodel.transform(df)
    println("ended lr's pip_data")

    return transform_df
  }

  def train(trainData:DataFrame,testData:DataFrame, n:Int)={
    val spark =trainData.sparkSession

    val pModel1=input.pip_fit(trainData,n)
    val new_train = input.genearate_new_column(trainData,"apply",n)
    val new_test = input.genearate_new_column(testData,"apply",n)
    val transform_train=pModel1.transform(new_train )   // fit
    val transform_test=pModel1.transform(new_test )
    //
    transform_train.cache()
    transform_test.cache()

    val (xgbModel,transform_train1,transform_test1)=xgboost.train(transform_train,transform_test,n)

    val leaf_train = xgboost.create_leaf(xgbModel,transform_train1)
    val leaf_test = xgboost.create_leaf(xgbModel,transform_test1)


    val pModel2= pip_fit(leaf_train,n)
    val transform_train2= pModel2.transform(leaf_train ).select("label","features2").cache()
    val transform_test2 = pModel2.transform(leaf_test  ).select("label","features2")


    val lr = new LogisticRegression()
      .setMaxIter(30)
      .setRegParam(1e-5)
      .setElasticNetParam(0)
      .setFeaturesCol("features2")
      .setLabelCol("label")
      .setStandardization(false)

    val lrModel=lr.fit(transform_train2)
    //println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // loss
    val trainingSummary = lrModel.summary
    println("loss:\n")
    trainingSummary.objectiveHistory.foreach(loss => println(loss))

    val predictions = lrModel.transform(transform_test2)
    //auc
    val scoreAndLabels2 = predictions.select( "probability","label").rdd.map{
      case Row(score: Vector, label: Double) => (score(1), label)
    }
    //      val scoreAndLabels2 = predictions.select( "prediction","label").rdd.map{
    //        t=>(t.getAs[Double]("prediction"),t.getAs[Double]("label"))
    //      }
    val auc2 = new BinaryClassificationMetrics(scoreAndLabels2).areaUnderROC
    println("Area under ROC:gbdt+LR = " + auc2)

    // save
    val fs = FileSystem.get(new Configuration())
    fs.delete(new Path( s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/lrModel" ), true)
    lrModel.save( s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/lrModel")

  }

  /**
   *
   * @param df
   * @param n 代表哪个模型
   */
  def predict_res(df:DataFrame, n:Int):DataFrame={
    val spark = df.sparkSession
    // create_leaf 这是瓶颈，吃内存特别厉害。容易内存不足。所以先对transform_apply进行廋身。
    // 用到的columns，
    val useColumns= Array("user_id","sku_id","features") ++: sparse_columns
    val transform_apply = input.pip_data(df ,n).select(useColumns.map(col(_)) : _*)
    //cache
//    transform_apply.persist(StorageLevel.DISK_ONLY)
    //load model
    val xgbModel  = XGBoost.loadModelFromHadoopFile(s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/_xgb")(spark.sparkContext)
    val leaf_apply = xgboost.create_leaf(xgbModel,transform_apply)


//    val transform_apply_tmp = transform_apply.select("user_id","sku_id", "features")
//    val leaf_apply_tmp = xgboost.create_leaf(xgbModel,transform_apply_tmp)
//    val add_df = transform_apply.select((Array("user_id","sku_id") ++: sparse_columns).map(col(_)) : _*)
//    val leaf_apply = leaf_apply_tmp.join(add_df,Seq("user_id","sku_id"))

    // 对结果进行廋身，因为内存不足
    val transform_apply2 =pip_data(leaf_apply ,n).select("user_id","sku_id", "features2")

    import spark.implicits._
    val lrModel = LogisticRegressionModel.load( s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/lrModel")
    val predictions =lrModel.transform(transform_apply2).select("user_id","sku_Id", "probability").rdd.map{
      case Row(user_id:String,sku_id:String,score: Vector) => (user_id,sku_id,score(1))
    }.toDF("user_id","sku_id", "prediction")

    predictions
  }


  def main(args: Array[String]) {
    val model_stage: String = args(0)
    val n:Int =args(1).toInt

    if (model_stage=="train"){
      val spark = SparkSession.builder()
        .appName("cpm.ltr.lr.train")
        .config("spark.rpc.askTimeout", "500")
        .config("spark.speculation", "false")
        .config("spark.memory.fraction", "0.8")
        .config("spark.memory.storageFraction", "0.5")
        .config("spark.shuffle.sort.bypassMergeThreshold","201")
        .enableHiveSupport()
        .getOrCreate()
      val (trainData,testData)=input.get_train_input(spark,n)

       train(trainData,testData,n)
    }

    else {
      val spark = SparkSession.builder()
        .appName("cpm.ltr.lr.predict")
        .config("spark.rpc.askTimeout", "500")
        .config("spark.speculation", "false")
        .config("spark.memory.fraction", "0.9")
        .config("spark.memory.storageFraction", "0.1")
        .config("spark.yarn.executor.memoryOverhead","8192")
        .enableHiveSupport()
        .getOrCreate()

      val apply_data=input.get_apply_input(spark,n) //.repartition(2000,col("user_id"))
      spark.sql("set spark.sql.shuffle.partitions = 2000")
      val predictions =predict_res(apply_data,n)

      val model_id:String = args(2)
      // top10
      val k=10
      predictions.createOrReplaceTempView("tmp_table")
      spark.sql("set spark.sql.shuffle.partitions =200")
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
