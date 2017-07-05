package com.jd.szad.LTR

/**
 * Created by xieliming on 2017/6/8.
 */

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object LR {

  def main(args: Array[String]) {
    val spark = SparkSession.builder()
      .appName("CPM.LTR.train")
      .config("spark.rpc.askTimeout", "500")
      .config("spark.speculation", "false")
      .config("spark.memory.fraction", "0.6")
      .config("spark.memory.storageFraction", "0.3")
      .enableHiveSupport()
      .getOrCreate()

    val model_stage: String = args(0)

    if (model_stage=="train"){
      val s1 =
        """
          |select Double(target_tag)  label
          |,nvl(sku_ctr_7                  ,0  )   sku_ctr_7
          |,nvl(sku_evr_7                  ,0  )   sku_evr_7
          |,nvl(sku_expose_nums_7          ,0  )   sku_expose_nums_7
          |,nvl(sku_click_nums_7           ,0  )   sku_click_nums_7
          |,nvl(sku_ad_gen_nums_7          ,0  )   sku_ad_gen_nums_7
          |,nvl(sku_ctr_15                 ,0  )   sku_ctr_15
          |,nvl(sku_evr_15                 ,0  )   sku_evr_15
          |,nvl(sku_expose_nums_15         ,0  )   sku_expose_nums_15
          |,nvl(sku_click_nums_15          ,0  )   sku_click_nums_15
          |,nvl(sku_ad_gen_nums_15         ,0  )   sku_ad_gen_nums_15
          |,nvl(cate_ctr_7                 ,0  )   cate_ctr_7
          |,nvl(cate_evr_7                 ,0  )   cate_evr_7
          |,nvl(cate_expose_nums_7         ,0  )   cate_expose_nums_7
          |,nvl(cate_click_nums_7          ,0  )   cate_click_nums_7
          |,nvl(cate_ad_gen_nums_7         ,0  )   cate_ad_gen_nums_7
          |,nvl(cate_ctr_15                ,0  )   cate_ctr_15
          |,nvl(cate_evr_15                ,0  )   cate_evr_15
          |,nvl(cate_expose_nums_15        ,0  )   cate_expose_nums_15
          |,nvl(cate_click_nums_15         ,0  )   cate_click_nums_15
          |,nvl(cate_ad_gen_nums_15        ,0  )   cate_ad_gen_nums_15
          |,nvl(sex                        ,'-1' )   sex
          |,nvl(buy_sex                    ,'-1' )   buy_sex
          |,nvl(age                        ,'-1' )   age
          |,nvl(carrer                     ,'-1' )   carrer
          |,nvl(marriage                   ,'-1' )   marriage
          |,nvl(haschild                   ,'-1' )   haschild
          |,nvl(province                   ,'-1' )   province
          |,nvl(city                       ,'-1' )   city
          |,nvl(jd_lev                     ,'-1' )   jd_lev
          |,nvl(catefav1                   ,'-1' )   catefav1
          |,nvl(catefav2                   ,'-1' )   catefav2
          |,nvl(catefav3                   ,'-1' )   catefav3
          |,nvl(user_act_gap               ,0  )   user_act_gap
          |,nvl(browse_sku_nums_7          ,0  )   browse_sku_nums_7
          |,nvl(gen_sku_nums_7             ,0  )   gen_sku_nums_7
          |,nvl(gen_sku_fee_7              ,0  )   gen_sku_fee_7
          |,nvl(browse_sku_nums_30         ,0  )   browse_sku_nums_30
          |,nvl(gen_sku_nums_30            ,0  )   gen_sku_nums_30
          |,nvl(gen_sku_fee_30             ,0  )   gen_sku_fee_30
          |,nvl(sku_browse_nums_7          ,0  )   sku_browse_nums_7
          |,nvl(sku_browse_nums_rate_7     ,0  )   sku_browse_nums_rate_7
          |,nvl(sku_browse_uv_7            ,0  )   sku_browse_uv_7
          |,nvl(sku_browse_uv_rate_7       ,0  )   sku_browse_uv_rate_7
          |,nvl(sku_gen_nums_30            ,0  )   sku_gen_nums_30
          |,nvl(sku_gen_nums_rate_30       ,0  )   sku_gen_nums_rate_30
          |,nvl(sku_gen_fee_30             ,0  )   sku_gen_fee_30
          |,nvl(sku_gen_fee_rate_30        ,0  )   sku_gen_fee_rate_30
          |,nvl(sku_comment_nums           ,0  )   sku_comment_nums
          |,nvl(sku_comment_score          ,0  )   sku_comment_score
          |,nvl(sku_comment_good_nums      ,0  )   sku_comment_good_nums
          |,nvl(sku_comment_good_rate      ,0  )   sku_comment_good_rate
          |,nvl(sku_comment_bad_nums       ,0  )   sku_comment_bad_nums
          |,nvl(sku_comment_bad_rate       ,0  )   sku_comment_bad_rate
          |,nvl(sku_hotscore               ,0  )   sku_hotscore
          |,nvl(sku_cvr                    ,0  )   sku_cvr
          |,nvl(sku_jd_prc                 ,0  )   sku_jd_prc
          |,nvl(sku_jd_prc_after           ,0  )   sku_jd_prc_after
          |,nvl(sku_jd_prc_rate            ,0  )   sku_jd_prc_rate
          |,nvl(cate_id                    ,0  )   cate_id
          |,nvl(cate_browse_nums_7         ,0  )   cate_browse_nums_7
          |,nvl(cate_browse_uv_7           ,0  )   cate_browse_uv_7
          |,nvl(cate_gen_nums_30           ,0  )   cate_gen_nums_30
          |,nvl(cate_gen_fee_30            ,0  )   cate_gen_fee_30
          |,nvl(brand_id                   ,0  )   brand_id
          |,nvl(brand_browse_nums_7        ,0  )   brand_browse_nums_7
          |,nvl(brand_browse_uv_7          ,0  )   brand_browse_uv_7
          |,nvl(brand_gen_nums_30          ,0  )   brand_gen_nums_30
          |,nvl(brand_gen_fee_30           ,0  )   brand_gen_fee_30
          |,nvl(flag_browse_sku_7          ,0  )   flag_browse_sku_7
          |,nvl(flag_browse_cate_7         ,0  )   flag_browse_cate_7
          |,nvl(flag_browse_brand_7        ,0  )   flag_browse_brand_7
          |,nvl(flag_browse_sku_15         ,0  )   flag_browse_sku_15
          |,nvl(flag_browse_cate_15        ,0  )   flag_browse_cate_15
          |,nvl(flag_browse_brand_15       ,0  )   flag_browse_brand_15
          |,nvl(flag_browse_sku_30         ,0  )   flag_browse_sku_30
          |,nvl(flag_browse_cate_30        ,0  )   flag_browse_cate_30
          |,nvl(flag_browse_brand_30       ,0  )   flag_browse_brand_30
          |,nvl(flag_gen_sku_30            ,0  )   flag_gen_sku_30
          |,nvl(flag_gen_cate_30           ,0  )   flag_gen_cate_30
          |,nvl(flag_gen_brand_30          ,0  )   flag_gen_brand_30
          |,nvl(flag_gen_sku_90            ,0  )   flag_gen_sku_90
          |,nvl(flag_gen_cate_90           ,0  )   flag_gen_cate_90
          |,nvl(flag_gen_brand_90          ,0  )   flag_gen_brand_90
          |,nvl(browse_sku_gap             ,0  )   browse_sku_gap
          |,nvl(browse_cate_gap            ,0  )   browse_cate_gap
          |,nvl(browse_brand_gap           ,0  )   browse_brand_gap
          |,nvl(gen_sku_gap                ,0  )   gen_sku_gap
          |,nvl(gen_cate_gap               ,0  )   gen_cate_gap
          |,nvl(gen_brand_gap              ,0  )   gen_brand_gap
          | from app.app_szad_m_dyrec_rank_train
          | where user_type=1
        """.stripMargin

      val data = spark.sql(s1).repartition(10)

      // sampling
      val df0 = data.where(col("target_tag")===0)
      val df1 = data.where(col("target_tag")===1)
      val sample_data= df0.sample(false,0.2).union(df1)

      // feature
      val sex_indexer = new StringIndexer().setInputCol("sex").setOutputCol("index_sex").setHandleInvalid("skip")
      val buy_sex_indexer = new StringIndexer().setInputCol("buy_sex").setOutputCol("index_buy_sex")
      val age_indexer = new StringIndexer().setInputCol("age").setOutputCol("index_age").setHandleInvalid("skip")
      val carrer_indexer = new StringIndexer().setInputCol("carrer").setOutputCol("index_carrer").setHandleInvalid("skip")
      val marriage_indexer = new StringIndexer().setInputCol("marriage").setOutputCol("index_marriage").setHandleInvalid("skip")
      val haschild_indexer = new StringIndexer().setInputCol("haschild").setOutputCol("index_haschild").setHandleInvalid("skip")
      val province_indexer = new StringIndexer().setInputCol("province").setOutputCol("index_province").setHandleInvalid("skip")
      val city_indexer = new StringIndexer().setInputCol("city").setOutputCol("index_city").setHandleInvalid("skip")
      val jd_lev_indexer = new StringIndexer().setInputCol("jd_lev").setOutputCol("index_jd_lev").setHandleInvalid("skip")
      val catefav1_indexer = new StringIndexer().setInputCol("catefav1").setOutputCol("index_catefav1").setHandleInvalid("skip")
      val catefav2_indexer = new StringIndexer().setInputCol("catefav2").setOutputCol("index_catefav2").setHandleInvalid("skip")
      val catefav3_indexer = new StringIndexer().setInputCol("catefav3").setOutputCol("index_catefav3").setHandleInvalid("skip")

      //onehot
      val sex_encoder = new OneHotEncoder().setInputCol("index_sex").setOutputCol("vec_sex")
      val buy_sex_encoder = new OneHotEncoder().setInputCol("index_buy_sex").setOutputCol("vec_buy_sex")
      val age_encoder = new OneHotEncoder().setInputCol("index_age").setOutputCol("vec_age")
      val carrer_encoder = new OneHotEncoder().setInputCol("index_carrer").setOutputCol("vec_carrer")
      val marriage_encoder = new OneHotEncoder().setInputCol("index_marriage").setOutputCol("vec_marriage")
      val haschild_encoder = new OneHotEncoder().setInputCol("index_haschild").setOutputCol("vec_haschild")
      val province_encoder = new OneHotEncoder().setInputCol("index_province").setOutputCol("vec_province")
      val city_encoder = new OneHotEncoder().setInputCol("index_city").setOutputCol("vec_city")
      val jd_lev_encoder = new OneHotEncoder().setInputCol("index_jd_lev").setOutputCol("vec_jd_lev")
      val catefav1_encoder = new OneHotEncoder().setInputCol(catefav1_indexer.getOutputCol).setOutputCol("vec_catefav1")
      val catefav2_encoder = new OneHotEncoder().setInputCol(catefav2_indexer.getOutputCol).setOutputCol("vec_catefav2")
      val catefav3_encoder = new OneHotEncoder().setInputCol(catefav2_indexer.getOutputCol).setOutputCol("vec_catefav3")


      //  Assembler
      val ColNames=data.schema.fieldNames
      val aa = Array("sex", "buy_sex", "age","carrer","marriage","haschild","province","city","jd_lev", "catefav1","catefav2","catefav3",
        "label","cate_id","brand_id")
      val bb = Array("vec_sex", "vec_buy_sex", "vec_age","vec_carrer","vec_marriage","vec_haschild","vec_province","vec_city","vec_jd_lev",
        "vec_catefav1","vec_catefav2","vec_catefav3")
      val VectorInputCols=(ColNames ++: bb).filter(!aa.contains(_)) //VectorInputCols.length //dropWhile不起作用,才用的filter
      val assembler = new VectorAssembler()
          .setInputCols(VectorInputCols)
          .setOutputCol("features")

      val pipeline = new Pipeline().setStages(Array(
        sex_indexer,buy_sex_indexer,age_indexer,carrer_indexer ,marriage_indexer ,haschild_indexer ,
        province_indexer,city_indexer,jd_lev_indexer,catefav1_indexer,catefav2_indexer,catefav3_indexer,
        sex_encoder,buy_sex_encoder,age_encoder,carrer_encoder ,marriage_encoder ,haschild_encoder ,
        province_encoder,city_encoder,jd_lev_encoder,catefav1_encoder,catefav2_encoder,catefav3_encoder, assembler)
      )

      val trainingData=sample_data
      val testData=data
      val pModel = pipeline.fit(trainingData)
      val transform_data=pModel.transform(trainingData).select("label","features");
      val transform_test=pModel.transform(testData).select("label","features")

      val model_type = "GBDT"
      if (model_type =="LR") {
        // model
        val lr = new LogisticRegression()
          .setMaxIter(30)
          .setRegParam(0.01)
          .setElasticNetParam(0)
          .setFeaturesCol("features")
          .setLabelCol("label")

        //    val lrModel = pModel.stages.last.asInstanceOf[LogisticRegressionModel]
        val lrModel = lr.fit(transform_data)
        println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}") // VectorInputCols(0)

        // 模型信息
        // loss
        val trainingSummary = lrModel.summary
        println("loss:")
        trainingSummary.objectiveHistory.foreach(loss => println(loss))

        val binarySummary = trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]
        val auc = binarySummary.areaUnderROC
        println(s"areaUnderROC: ${auc}")

        // eval
        val transform_test = pModel.transform(testData).select( "label", "features")
        val predictions = lrModel.transform(transform_test)
        predictions.select( "label", "features","probability","prediction").where(col("label")===1).show(5,false)
        // confusionMatrix
        predictions.groupBy("label","prediction").agg(count("label")).show()

        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("label")
          .setPredictionCol("prediction")
          .setMetricName("accuracy")
        val accuracy = evaluator.evaluate(predictions)
        println("Test Error = " + (1.0 - accuracy))

      }else if (model_type =="GBDT"){

        // GBTClassifier or  GBTRegressor
        // val gbt=new GBTClassifier() .setFeaturesCol("features").setLabelCol("label").setMaxIter(20)
        val gbt =new GBTRegressor()
          .setFeaturesCol("features")
          .setLabelCol("label")
          .setMaxIter(10)
          .setLossType("squared")

        val gdtModel = gbt.fit(transform_data)
        println(" GBT model:\n" + gdtModel.toDebugString)
        println(" GBT featureImportances:\n" + gdtModel.featureImportances)  // VectorInputCols(20)

        //predition
        val gdt_predictions = gdtModel.transform(transform_test)

        //auc
        val scoreAndLabels = gdt_predictions.select( "prediction","label").rdd.map{
          t=>(t.getAs("prediction").asInstanceOf[Double],t.getAs("label").asInstanceOf[Double])
        }
        val BinaryMetrics=new BinaryClassificationMetrics(scoreAndLabels)
        val auc = BinaryMetrics.areaUnderROC
        println("Area under ROC = " + auc)

        //save
        val fsconf = new Configuration()
        val fs = FileSystem.get(fsconf)
        fs.delete(new Path( "ads_sz/app.db/app_szad_m_dyrec_rank_model_gbt" ), true)
        fs.delete(new Path( "ads_sz/app.db/app_szad_m_dyrec_rank_model_pipline" ), true)

        gdtModel.save("ads_sz/app.db/app_szad_m_dyrec_rank_model_gbt")
        pModel.save("ads_sz/app.db/app_szad_m_dyrec_rank_model_pipline")
      }

    }

    else {
      // load model
      val gdtModel = GBTRegressionModel.load("ads_sz/app.db/app_szad_m_dyrec_rank_model_gbt")
      val pmodel = PipelineModel.load("ads_sz/app.db/app_szad_m_dyrec_rank_model_pipline")

      // apply data
      val s2 =
        """
          |select Double(target_tag)  label
          |,user_id
          |,sku_Id
          |,nvl(sku_ctr_7                  ,0  )   sku_ctr_7
          |,nvl(sku_evr_7                  ,0  )   sku_evr_7
          |,nvl(sku_expose_nums_7          ,0  )   sku_expose_nums_7
          |,nvl(sku_click_nums_7           ,0  )   sku_click_nums_7
          |,nvl(sku_ad_gen_nums_7          ,0  )   sku_ad_gen_nums_7
          |,nvl(sku_ctr_15                 ,0  )   sku_ctr_15
          |,nvl(sku_evr_15                 ,0  )   sku_evr_15
          |,nvl(sku_expose_nums_15         ,0  )   sku_expose_nums_15
          |,nvl(sku_click_nums_15          ,0  )   sku_click_nums_15
          |,nvl(sku_ad_gen_nums_15         ,0  )   sku_ad_gen_nums_15
          |,nvl(cate_ctr_7                 ,0  )   cate_ctr_7
          |,nvl(cate_evr_7                 ,0  )   cate_evr_7
          |,nvl(cate_expose_nums_7         ,0  )   cate_expose_nums_7
          |,nvl(cate_click_nums_7          ,0  )   cate_click_nums_7
          |,nvl(cate_ad_gen_nums_7         ,0  )   cate_ad_gen_nums_7
          |,nvl(cate_ctr_15                ,0  )   cate_ctr_15
          |,nvl(cate_evr_15                ,0  )   cate_evr_15
          |,nvl(cate_expose_nums_15        ,0  )   cate_expose_nums_15
          |,nvl(cate_click_nums_15         ,0  )   cate_click_nums_15
          |,nvl(cate_ad_gen_nums_15        ,0  )   cate_ad_gen_nums_15
          |,nvl(sex                        ,'-1' )   sex
          |,nvl(buy_sex                    ,'-1' )   buy_sex
          |,nvl(age                        ,'-1' )   age
          |,nvl(carrer                     ,'-1' )   carrer
          |,nvl(marriage                   ,'-1' )   marriage
          |,nvl(haschild                   ,'-1' )   haschild
          |,nvl(province                   ,'-1' )   province
          |,nvl(city                       ,'-1' )   city
          |,nvl(jd_lev                     ,'-1' )   jd_lev
          |,nvl(catefav1                   ,'-1' )   catefav1
          |,nvl(catefav2                   ,'-1' )   catefav2
          |,nvl(catefav3                   ,'-1' )   catefav3
          |,nvl(user_act_gap               ,0  )   user_act_gap
          |,nvl(browse_sku_nums_7          ,0  )   browse_sku_nums_7
          |,nvl(gen_sku_nums_7             ,0  )   gen_sku_nums_7
          |,nvl(gen_sku_fee_7              ,0  )   gen_sku_fee_7
          |,nvl(browse_sku_nums_30         ,0  )   browse_sku_nums_30
          |,nvl(gen_sku_nums_30            ,0  )   gen_sku_nums_30
          |,nvl(gen_sku_fee_30             ,0  )   gen_sku_fee_30
          |,nvl(sku_browse_nums_7          ,0  )   sku_browse_nums_7
          |,nvl(sku_browse_nums_rate_7     ,0  )   sku_browse_nums_rate_7
          |,nvl(sku_browse_uv_7            ,0  )   sku_browse_uv_7
          |,nvl(sku_browse_uv_rate_7       ,0  )   sku_browse_uv_rate_7
          |,nvl(sku_gen_nums_30            ,0  )   sku_gen_nums_30
          |,nvl(sku_gen_nums_rate_30       ,0  )   sku_gen_nums_rate_30
          |,nvl(sku_gen_fee_30             ,0  )   sku_gen_fee_30
          |,nvl(sku_gen_fee_rate_30        ,0  )   sku_gen_fee_rate_30
          |,nvl(sku_comment_nums           ,0  )   sku_comment_nums
          |,nvl(sku_comment_score          ,0  )   sku_comment_score
          |,nvl(sku_comment_good_nums      ,0  )   sku_comment_good_nums
          |,nvl(sku_comment_good_rate      ,0  )   sku_comment_good_rate
          |,nvl(sku_comment_bad_nums       ,0  )   sku_comment_bad_nums
          |,nvl(sku_comment_bad_rate       ,0  )   sku_comment_bad_rate
          |,nvl(sku_hotscore               ,0  )   sku_hotscore
          |,nvl(sku_cvr                    ,0  )   sku_cvr
          |,nvl(sku_jd_prc                 ,0  )   sku_jd_prc
          |,nvl(sku_jd_prc_after           ,0  )   sku_jd_prc_after
          |,nvl(sku_jd_prc_rate            ,0  )   sku_jd_prc_rate
          |,nvl(cate_id                    ,0  )   cate_id
          |,nvl(cate_browse_nums_7         ,0  )   cate_browse_nums_7
          |,nvl(cate_browse_uv_7           ,0  )   cate_browse_uv_7
          |,nvl(cate_gen_nums_30           ,0  )   cate_gen_nums_30
          |,nvl(cate_gen_fee_30            ,0  )   cate_gen_fee_30
          |,nvl(brand_id                   ,0  )   brand_id
          |,nvl(brand_browse_nums_7        ,0  )   brand_browse_nums_7
          |,nvl(brand_browse_uv_7          ,0  )   brand_browse_uv_7
          |,nvl(brand_gen_nums_30          ,0  )   brand_gen_nums_30
          |,nvl(brand_gen_fee_30           ,0  )   brand_gen_fee_30
          |,nvl(flag_browse_sku_7          ,0  )   flag_browse_sku_7
          |,nvl(flag_browse_cate_7         ,0  )   flag_browse_cate_7
          |,nvl(flag_browse_brand_7        ,0  )   flag_browse_brand_7
          |,nvl(flag_browse_sku_15         ,0  )   flag_browse_sku_15
          |,nvl(flag_browse_cate_15        ,0  )   flag_browse_cate_15
          |,nvl(flag_browse_brand_15       ,0  )   flag_browse_brand_15
          |,nvl(flag_browse_sku_30         ,0  )   flag_browse_sku_30
          |,nvl(flag_browse_cate_30        ,0  )   flag_browse_cate_30
          |,nvl(flag_browse_brand_30       ,0  )   flag_browse_brand_30
          |,nvl(flag_gen_sku_30            ,0  )   flag_gen_sku_30
          |,nvl(flag_gen_cate_30           ,0  )   flag_gen_cate_30
          |,nvl(flag_gen_brand_30          ,0  )   flag_gen_brand_30
          |,nvl(flag_gen_sku_90            ,0  )   flag_gen_sku_90
          |,nvl(flag_gen_cate_90           ,0  )   flag_gen_cate_90
          |,nvl(flag_gen_brand_90          ,0  )   flag_gen_brand_90
          |,nvl(browse_sku_gap             ,0  )   browse_sku_gap
          |,nvl(browse_cate_gap            ,0  )   browse_cate_gap
          |,nvl(browse_brand_gap           ,0  )   browse_brand_gap
          |,nvl(gen_sku_gap                ,0  )   gen_sku_gap
          |,nvl(gen_cate_gap               ,0  )   gen_cate_gap
          |,nvl(gen_brand_gap              ,0  )   gen_brand_gap
          | from app.app_szad_m_dyrec_rank_apply
          | where user_type=1
        """.stripMargin

      val apply_data = spark.sql(s2)
      val transofrm_apply = pmodel.transform(apply_data)
      val predictions = gdtModel.transform(transofrm_apply ).select("user_id","sku_Id", "prediction")

      // top10
      val k=10

      predictions.createOrReplaceTempView("tmp_table")
      spark.sql("use app")
      spark.sql("set mapreduce.output.fileoutputformat.compress=true")
      spark.sql("set hive.exec.compress.output=true")
      spark.sql("set mapred.output.compression.codec=com.hadoop.compression.lzo.LzopCodec")
      val ss =
        s"""
          |insert overwrite table app.app_szad_m_dyrec_rank_predict_res partition(model_id=3,user_type=1)
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
