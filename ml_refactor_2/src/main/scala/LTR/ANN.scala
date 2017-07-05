package com.jd.szad.LTR

/**
 * Created by xieliming on 2017/6/8.
 */

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.feature.{StandardScaler, OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._


object ANN {

  def main(args: Array[String]) {
    val spark = SparkSession.builder()
      .appName("CPM.LTR.ANN")
      .config("spark.rpc.askTimeout", "500")
      .config("spark.speculation", "false")
      .config("spark.memory.fraction", "0.6")
      .config("spark.memory.storageFraction", "0.3")
      .enableHiveSupport()
      .getOrCreate()

    val model_type :String = args(0)
    val model_stage :String = args(1)


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
    val sample_data= df0.sample(false,0.1).union(df1)

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
    val aa = Array("sex", "buy_sex", "age","carrer","marriage","haschild","province","city","jd_lev", "catefav1","catefav2","catefav3")
    val bb = Array("vec_sex", "vec_buy_sex", "vec_age","vec_carrer","vec_marriage","vec_haschild","vec_province","vec_city","vec_jd_lev")
    val VectorInputCols=(ColNames ++: bb).filter(!aa.contains(_)).drop(1) //VectorInputCols.length //dropWhile不起作用,才用的filter
    val assembler = new VectorAssembler()
      .setInputCols(VectorInputCols)
      .setOutputCol("features")

    // norm
    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(true)

    val pipeline = new Pipeline().setStages(Array(
      sex_indexer,buy_sex_indexer,age_indexer,carrer_indexer ,marriage_indexer ,haschild_indexer , province_indexer,city_indexer,jd_lev_indexer ,
      sex_encoder,buy_sex_encoder,age_encoder,carrer_encoder ,marriage_encoder ,haschild_encoder , province_encoder,city_encoder,jd_lev_encoder , assembler,scaler)
    )

    val trainingData=sample_data
    val testData=data
    val pModel = pipeline.fit(trainingData)
    val transform_data=pModel.transform(trainingData).select("label","features")
    val transform_test=pModel.transform(testData).select("label","features")


    val numAttrs = transform_data.first().getAs[Vector]("features").size
    val layers = Array[Int](numAttrs, 1000,  2)

    // create the trainer and set its parameters
    val ANN = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(1000)  //batch_size
      .setSeed(1234L)
      .setMaxIter(20)
      .setSolver("gd").setStepSize(0.5)

    val Model = ANN.fit(transform_data)


    //predition
    val predictions = Model.transform(transform_data)

    //auc
    val scoreAndLabels = predictions.select( "prediction","label").rdd.map{
      t=>(t.getAs("prediction").asInstanceOf[Double],t.getAs("label").asInstanceOf[Double])
    }
    val BinaryMetrics=new BinaryClassificationMetrics(scoreAndLabels)
    val auc = BinaryMetrics.areaUnderROC
    println("Area under ROC = " + auc)

    // confusionMatrix
    predictions.groupBy("label","prediction").agg(count("label")).show()


  }



}
