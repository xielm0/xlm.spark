package com.jd.szad.LTR

import ml.dmlc.xgboost4j.scala.spark.XGBoost
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Row, SparkSession}

/**
 * Created by xieliming on 2017/7/5.
 */
object xgboost {
  def main(args: Array[String]) {
    val spark = SparkSession.builder()
      .appName("CPM.xgboost")
      .config("spark.rpc.askTimeout", "500")
      .config("spark.speculation", "false")
      .config("spark.memory.fraction", "0.6")
      .config("spark.memory.storageFraction", "0.3")
      .enableHiveSupport()
      .getOrCreate()

    val model_stage: String = args(0)

    if (model_stage == "train") {
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
      val df0 = data.where(col("target_tag") === 0)
      val df1 = data.where(col("target_tag") === 1)
      val sample_data = df0.sample(false, 0.2).union(df1)

      // feature
      val stringColumns = Array("sex", "buy_sex", "age", "carrer", "marriage", "haschild", "province", "city", "jd_lev",
        "catefav1", "catefav2", "catefav3", "cate_id", "brand_id")

      val string_index: Array[org.apache.spark.ml.PipelineStage] = stringColumns.map(
        cname => new StringIndexer()
          .setInputCol(cname)
          .setOutputCol(s"${cname}_index").setHandleInvalid("skip")
      )

      val indexColumns = Array("sex_index", "buy_sex_index", "age_index", "carrer_index", "marriage_index", "haschild_index", "province_index", "city_index", "jd_lev_index",
        "catefav1_index", "catefav2_index", "catefav3_index", "cate_id_index", "brand_id_index")
      val Onehot_Encoder: Array[org.apache.spark.ml.PipelineStage] = indexColumns.map(
        cname => new OneHotEncoder()
          .setInputCol(cname)
          .setOutputCol(s"${cname}_vec")
      )

      val vecColumns = Array("sex_index_vec", "buy_sex_index_vec", "age_index_vec", "carrer_index_vec", "marriage_index_vec", "haschild_index_vec",
        "province_index_vec", "city_index", "jd_lev_index",
        "catefav1_index_vec", "catefav2_index_vec", "catefav3_index_vec")

      //  Assembler
      val AllColumns = data.schema.fieldNames
      val aa = Array("sex", "buy_sex", "age", "carrer", "marriage", "haschild", "province", "city", "jd_lev", "catefav1", "catefav2", "catefav3",
        "label", "cate_id", "brand_id")

      val VectorInputCols = (AllColumns ++: vecColumns).filter(!aa.contains(_)) //VectorInputCols.length //dropWhile不起作用,才用的filter
      val assembler = new VectorAssembler()
          .setInputCols(VectorInputCols)
          .setOutputCol("features")

      val pipeline = new Pipeline().setStages(string_index ++: Onehot_Encoder ++: Array(assembler))

      val trainingData = sample_data
      val testData = data
      val pModel = pipeline.fit(trainingData)
      val transform_data = pModel.transform(trainingData).select("label", "features");
      val transform_test = pModel.transform(testData).select("label", "features")

    //xgboost
    val paramMap = List(
      "eta" -> 0.1f,
      "max_depth" -> 5,
      "objective" -> "binary:logistic").toMap

    val numRound = 20
    val xgbModel = XGBoost.trainWithDataFrame( transform_data, paramMap, numRound, nWorkers=10 ,  useExternalMemory = true )
      xgbModel.getParam("objective")

      //predition
      val xgb_predictions = xgbModel.transform(transform_test)

      //auc
      import org.apache.spark.ml.linalg.Vector
      val scoreAndLabels = xgb_predictions.select( "probabilities","label").rdd.map{
        case Row(score: Vector, label: Double) => (score(1), label)
      }
      val predictAndLabels = xgb_predictions.select( "prediction","label").rdd.map{
        t=>(t.getAs("prediction").asInstanceOf[Double],t.getAs("label").asInstanceOf[Double])
      }
      val BinaryMetrics=new BinaryClassificationMetrics(scoreAndLabels)
      val auc = BinaryMetrics.areaUnderROC
      println("Area under ROC = " + auc)


    }


  }
}
