package LTR

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql._

/**
 * Created by xieliming on 2017/9/18.
 */
object gbdt {
  val num_round=20
  def train(transform_train:DataFrame,transform_test:DataFrame,n:Int):(DataFrame,DataFrame)={
    transform_train.cache()
    transform_test.cache()
    //
    val gbt =new GBTRegressor()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setMaxIter(20)
      .setLossType("squared")

    val gdtModel = gbt.fit(transform_train)
    println(" GBT model:\n" + gdtModel.toDebugString)
    println(" GBT feature Importances:\n" + gdtModel.featureImportances)  // VectorInputCols(20)
    //save
    val fs = FileSystem.get(new Configuration())
    fs.delete(new Path( s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/gbdt" ), true)
    gdtModel.save(s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/gbdt")


    //predition
    val transform_train1 = gdtModel.transform(transform_train).withColumnRenamed("prediction","gdt_predication")
    val transform_test1  = gdtModel.transform(transform_test ).withColumnRenamed("prediction","gdt_predication")

    //auc
    val scoreAndLabels = transform_test1.select( "gdt_predication","label").rdd.map{
      t=>(t.getAs[Double]("gdt_predication"),t.getAs[Double]("label"))
    }
    val BinaryMetrics=new BinaryClassificationMetrics(scoreAndLabels)
    val auc = BinaryMetrics.areaUnderROC
    println("Area under ROC = " + auc)


    transform_train.unpersist()
    transform_test.unpersist()
    return(transform_train1,transform_test1)


  }


  def predict_res(transform_df:DataFrame,n:Int):DataFrame={
    // load
    // load model
    val gdtModel = GBTRegressionModel.load(s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/gbdt" )
    val res=gdtModel.transform(transform_df).withColumnRenamed("prediction","gdt_predication")
    res

  }


  def main(args: Array[String]) {
    val spark = SparkSession.builder()
      .appName("CPM.LTR.GBDT")
      .config("spark.rpc.askTimeout", "500")
      .config("spark.speculation", "false")
      .config("spark.memory.fraction", "0.75")
      .config("spark.memory.storageFraction", "0.5")
      .enableHiveSupport()
      .getOrCreate()

    val n=7
    val (trainData, testData) = input.get_train_input(spark,n)
    val pModel=input.pip_fit(trainData,n)
    val new_train = input.genearate_new_column(trainData,"apply",n)
    val new_test = input.genearate_new_column(testData,"apply",n)
    val transform_train = pModel.transform(new_train )
    val transform_test = pModel.transform(new_test )
    train(transform_train, transform_test,n)

  }

}
