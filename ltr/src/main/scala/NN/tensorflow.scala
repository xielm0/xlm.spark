package NN

import LTR.input
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession

/**
  * Created by xieliming on 2017/5/16.
  * spark-shell --num-executors 5 --executor-memory 8g --executor-cores 4
  */
object tensorflow {

   def main(args: Array[String]) {
     val spark = SparkSession.builder()
       .appName("cpm.ltr.tensor")
       .config("spark.rpc.askTimeout", "500")
       .config("spark.speculation", "false")
       .config("spark.memory.fraction", "0.6")
       .config("spark.memory.storageFraction", "0.3")
       .enableHiveSupport()
       .getOrCreate()

     val train_flag:String = args(0)
     val dt_from:String = args(1)
     val dt_to:String = args(2)
     val n =7

     if (train_flag =="train") {
       val s1 =
         s"""
            |select Double(target_tag)  label
            |      ,sku_id
            |      ,nvl (sku_ctr                    ,0    )     sku_ctr
            |      ,nvl (sku_ctr_up                 ,0    )     sku_ctr_up
            |      ,nvl (sku_ctr_low                ,0    )     sku_ctr_low
            |      ,nvl (sku_ctr_mid                ,0    )     sku_ctr_mid
            |      ,nvl (sku_ctr_gap                ,0    )     sku_ctr_gap
            |      ,nvl (sku_browse_nums            ,0    )     sku_browse_nums
            |      ,nvl (sku_gen_nums               ,0    )     sku_gen_nums
            |      ,nvl (sku_comment_nums           ,0    )     sku_comment_nums
            |      ,nvl (sku_comment_score          ,0    )     sku_comment_score
            |      ,nvl (sku_comment_good_nums      ,0    )     sku_comment_good_nums
            |      ,nvl (sku_comment_good_rate      ,0    )     sku_comment_good_rate
            |      ,nvl (sku_comment_bad_nums       ,0    )     sku_comment_bad_nums
            |      ,nvl (sku_comment_bad_rate       ,0    )     sku_comment_bad_rate
            |      ,nvl (sku_hotscore               ,0    )     sku_hotscore
            |      ,nvl (sku_cvr                    ,0    )     sku_cvr
            |      ,nvl (sku_jd_prc                 ,0    )     sku_jd_prc
            |      ,nvl (sku_jd_prc_after           ,0    )     sku_jd_prc_after
            |      ,nvl (sku_jd_prc_rate            ,0    )     sku_jd_prc_rate
            |      ,nvl (string(cate_id )           ,'s'  )     cate_id
            |      ,nvl (string(brand_id )          ,'s'  )     brand_id
            |      ,nvl (sex                        ,'s'  )     sex
            |      ,nvl (age                        ,'s'  )     age
            |      ,nvl (carrer                     ,'s'  )     carrer
            |      ,nvl (marriage                   ,'s'  )     marriage
            |      ,nvl (haschild                   ,'s'  )     haschild
            |      ,nvl (province                   ,'s'  )     province
            |      ,nvl (city                       ,'s'  )     city
            |      ,nvl (jd_lev                     ,'s'  )     jd_lev
            |      ,nvl (browse_list_sku_date       ,'s:999'  )     browse_list_sku_date
            |      ,nvl (search_list_sku_date       ,'s:999'  )     search_list_sku_date
            |      ,nvl (fav_list_sku_date          ,'s:999'  )     fav_list_sku_date
            |      ,nvl (car_list_sku_date          ,'s:999'  )     car_list_sku_date
            |      ,nvl (gen_list_sku_date          ,'s:999'  )     gen_list_sku_date
            |      ,nvl (browse_list_cate_date      ,'s:999'  )     browse_list_cate_date
            |      ,nvl (search_list_cate_date      ,'s:999'  )     search_list_cate_date
            |      ,nvl (fav_list_cate_date         ,'s:999'  )     fav_list_cate_date
            |      ,nvl (car_list_cate_date         ,'s:999'  )     car_list_cate_date
            |      ,nvl (gen_list_cate_date         ,'s:999'  )     gen_list_cate_date
            |  from app.app_szad_m_dyrec_rank_train_new
            | where user_type=1
            |   and dt>'${dt_from}' and dt<='${dt_to}'
            |   and n=${n}
            |   and user_id is not null
            |   and sku_id is not null
      """.stripMargin

       val data = spark.sql(s1)

       // 生成新的列
       val new_df =input.genearate_new_column(data,"apply",n)



       // 数据处理
       val stringColumns = input.id_columns_thin ++ input.str_columns_direct ++ input.str_columns_new  ++ input.cross_columns_tuple_s
       val stringColumns_ind=stringColumns.map(_+"_ind")

       val string_index: Array[org.apache.spark.ml.PipelineStage] = stringColumns.map(
         cname => new StringIndexer()
           .setInputCol(cname)
           .setOutputCol(s"${cname}_ind").setHandleInvalid("skip")
       )



       //  Assembler
       val AllColumns = data.schema.fieldNames
       val addColumns = input.flag_columns ++ stringColumns_ind
       val delColumns =  input.id_columns ++: input.str_columns_del ++: input.str_columns_direct  ++: Array("label")


       // 这里不用vecColumns
       val VectorInputCols = (AllColumns ++: addColumns).filter(!delColumns.contains(_)) //VectorInputCols.length //dropWhile不起作用,才用的filter
       //将字段打印出来
       println("VectorInputCols.length="+ VectorInputCols.length)
       println("VectorInputCols.columns="+VectorInputCols.mkString(","))

       val assembler = new VectorAssembler().setInputCols(VectorInputCols).setOutputCol("features")

       // norm
       // val scaler = new StandardScaler() .setInputCol("features") .setOutputCol("scaledFeatures") .setWithStd(true) .setWithMean(true)

       //val pipeline = new Pipeline().setStages(string_index  ++: Onehot_Encoder ++: Array(assembler,scaler))
       val pipeline = new Pipeline().setStages(string_index  ++: Array(assembler))
       val pModel = pipeline.fit(new_df)

       val transform_data=pModel.transform(new_df).select("label","features")
       val rdd_data=transform_data.rdd.map { t =>
         val label = t.getAs[Double]("label")
         val vec = t.getAs[Vector]("features")
         label +"," + vec.toArray.mkString(",")
       }
       val splits = rdd_data.randomSplit(Array(0.8,0.2),12345)
       val (transform_train,transform_test)= (splits(0), splits(1))

       //save
       val fs = FileSystem.get(new Configuration())
       fs.delete(new Path( s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/tensorflow_pip" ),true)
       pModel.save( s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/tensorflow_pip")

       fs.delete(new Path( "ads_sz/app.db/app_szad_m_dyrec_rank_train_tensorflow/type=train_new" ),true)
       fs.delete(new Path( "ads_sz/app.db/app_szad_m_dyrec_rank_train_tensorflow/type=test_new" ),true)


       transform_train.repartition(20).saveAsTextFile("ads_sz/app.db/app_szad_m_dyrec_rank_train_tensorflow/type=train_new" )
       transform_test.repartition(20).saveAsTextFile("ads_sz/app.db/app_szad_m_dyrec_rank_train_tensorflow/type=test_new" )
     }
     else if(train_flag =="apply"){

       val data = input.get_apply_input(spark,7)
       val new_df =input.genearate_new_column(data,"apply",n)

       val pModel = PipelineModel.load( s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/tensorflow_pip")

       val transform_data=pModel.transform(new_df).select("user_id","sku_id","label","features")
       val rdd_data=transform_data.rdd.map { t =>
         val label = t.getAs[Double]("label")
         val vec = t.getAs[Vector]("features")
         t.getAs[String]("user_id")+ "," + t.getAs[String]("sku_id") + "," + label +"," + vec.toArray.mkString(",")
       }


       //save
       val fs = FileSystem.get(new Configuration())
       fs.delete(new Path( "ads_sz/app.db/app_szad_m_dyrec_rank_train_tensorflow/type=apply_new" ),true)
       //
       rdd_data.saveAsTextFile("ads_sz/app.db/app_szad_m_dyrec_rank_train_tensorflow/type=apply_new" )

     }


   }
 }
