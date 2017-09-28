package LTR

import java.text.SimpleDateFormat
import java.util.Calendar

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

import scala.collection.mutable.ArrayBuffer

/**
 * Created by xieliming on 2017/9/14.
 */
object input {

  /**
   * getDate(0,"yyyy-MM-dd")
   * getDate(-1,"yyyy-MM-dd HH:mm:ss")
   */
  def getDate(interval: Int=0,time_format:String ="yyyy-MM-dd"):String={
    val myFmt:SimpleDateFormat = new SimpleDateFormat(time_format)
    val cal:java.util.Calendar = Calendar.getInstance()
    // 当前时间
    // val cur_time:java.util.Date = cal.getTime()
    // 加减n天
    cal.add(Calendar.DATE, interval)
    val time:java.util.Date=cal.getTime()
    myFmt.format(time)
  }

  val dt_to =getDate(-1,"yyyy-MM-dd")
  val dt_from =getDate(-8,"yyyy-MM-dd")

  def get_train_input(spark:SparkSession,n:Int): (DataFrame,DataFrame) ={
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

    val data = spark.sql(s1).repartition(100,col("sku_id")).cache()

    // sampling
//    val df0 = data.where(col("target_tag")===0)
//    val df1 = data.where(col("target_tag")===1)
//    val splits = df0.sample(false,0.2 , 12345).union(df1 ).randomSplit(Array(0.8,0.2),12345)
    val splits = data.randomSplit(Array(0.8,0.2),12345)
    val (trainData,testData)= (splits(0), splits(1))
    //    val trainingData= df0.sample(false,0.2).union(df1.sample(false,0.7))
    //    val testData=data
    println("trainData count = " + trainData.count())

    (trainData,testData)

  }


  def get_apply_input(spark:SparkSession,n:Int):DataFrame ={
    val s2 =
      s"""
         |select Double(target_tag)  label
         |      ,user_id
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
         |  from app.app_szad_m_dyrec_rank_apply_new
         | where user_type=1
         |   and n=${n}
         |   and user_id is not null
         |   and sku_id is not null
      """.stripMargin

    val data = spark.sql(s2)
    val cache_data=data

    cache_data

  }



  def udf_user_act_gap(list_sku_date:String):Int={
    val sku_date= list_sku_date.split("#")
    val date_list = sku_date.map(t=> t.split(":")(1).toInt)
    date_list.min
  }

  def udf_flag_browse_sku(list_sku_date:String,sku_id:String):Int={
    val sku_date= list_sku_date.split("#")
    val sku_list =  sku_date.map(t=> t.split(":")(0))
    val flag_browse_sku = if(sku_list.contains(sku_id)) 1 else 0
    flag_browse_sku
  }

  def udf_browse_sku_gap(list_sku_date:String,sku_id:String):Int={
    val sku_date= list_sku_date.split("#")
    val sku_list = sku_date.map(t=> t.split(":")(0))
    val date_list = sku_date.map(t=> t.split(":")(1).toInt)
    val browse_sku_gap = if(sku_list.contains(sku_id)){
      val i = sku_list.indexOf(sku_id)
      date_list(i)
    }else {999}
    browse_sku_gap
  }

  def udf_cross_sku1_sku2(list_sku_date:String,sku_id:String,distinct_flag:Int):Array[Int]={
    val sku_date= list_sku_date.split("#")
    val sku_list = if (distinct_flag==1) sku_date.map(t=> t.split(":")(0)).distinct else  sku_date.map(t=> t.split(":")(0))

    val hash_size = 1e5.toInt
    val k=10
    val cross_x = new Array[Int](k)
    val n = sku_list.size
    for (i <- Range(0,k,1)) {
      if (i<n) {
        cross_x(i)=  cross_s(sku_list(i),sku_id,hash_size)
      }else cross_x(i)=0
    }
    cross_x

  }

  def Array2Column( df: DataFrame,colname:String ,k:Int): DataFrame ={
    /**
     * 将array列，转变成多列。array的长度是k
     */
    var df_a:DataFrame=df
    var df_b:DataFrame=df

    for (i <- Range(0,k,1)) {
      df_b= df_a.withColumn(
        colname+"_"+i ,
        // 这里有个坑，ArrayType不是Array ,这里要用Seq
        udf( (cc: Seq[Int])  => cc(i).toDouble)
          .apply(col(colname))
      )
      df_a=df_b
    }
    val res=df_b
    res

  }

  /**
   *  统计某离散列的次数，取top k ，剩余的用默认值替代，加快运行速度。
   *  对sku_id, brand_id会用到
   *  要求该列是字符串
   *  这里不能用array,在apply时，array比较大时，比如1e5,就会导致array.contains很慢。
   */

  def thin_column(df:DataFrame,colName:String, k:Int, train_flag :String,n:Int):DataFrame={
    val new_colName=colName+"_thin"
    val spark = df.sparkSession

    if (train_flag=="fit"){
      val df1=df.groupBy(colName) .agg(count(colName) as "cnt").select(colName,"cnt").orderBy(col("cnt").desc)
      val topk = df1.take(k).map(t=> t.getAs(colName).toString +"," + "1" )
      // 转成rdd保存到hdfs
      val rdd1= spark.sparkContext.makeRDD(topk)
      //save
      val fs = FileSystem.get(new Configuration())
      fs.delete(new Path( s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/${colName}" ), true)
      rdd1.saveAsTextFile(s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/${colName}")

      val topk_map=topk.map(t=>(t.split(",")(0),t.split(",")(1) ) ).toMap
      val bc_k = spark.sparkContext.broadcast(topk_map)
      df.withColumn(new_colName,
        udf( (id:String)  => if (bc_k.value.contains(id ) ) id else "s" )
          .apply(col(colName))
      )
    }else {
      // load
      val rdd1 = spark.sparkContext.textFile(s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/${colName}")
      val topk_map= rdd1.map(t=> (t.split(",")(0),t.split(",")(1) ) ).collectAsMap()
      val bc_k = spark.sparkContext.broadcast(topk_map)

      df.withColumn(new_colName,
        udf( (id:String)  => if (bc_k.value.contains(id ) ) id else "s" )
          .apply(col(colName))
      )

    }
  }

  /**
   * case class Employee(id: Int, name: String)
   * val listOfEmployees = List(Employee(1, "iteblog"), Employee(2, "Jason"), Employee(3, "Abhi"))
   * val df = spark.sqlContext.createDataFrame(listOfEmployees)
   * cross_column(df,"id","name",1000).show()
   */
  def cross_s(a: String, b:String,hash_bucket_size:Int):Int={
    math.abs((a + ":" + b).hashCode) % hash_bucket_size
  }

  def cross_column(df:DataFrame, c1:String, c2:String ,hash_bucket_size:Int)={
    val new_colname= Array(c1,c2).mkString("_")
    val df2=df.withColumn(new_colname,
      udf( (a: String, b:String)  => math.abs((a + ":" + b).hashCode) % hash_bucket_size )
        .apply(col(c1),col(c2))
    )
    df2
  }

  val cross_columns_tuple=Array(Array("sex","cate_id"),Array("haschild","cate_id"),
    Array("user_act_gap","cate_id"), Array("browse_cate_gap","cate_id"), Array("gen_cate_gap","cate_id"))
  val cross_columns_tuple_s=cross_columns_tuple.map(t=>t.mkString("_"))

  val cross_column_tuple_index=cross_columns_tuple_s.map(t=>t+"_ind")

  val cross_column_tuple_vec=cross_column_tuple_index.map(t=>t+"_vec")

  def cross_column(df:DataFrame):DataFrame={
    var df_a =df
    var df_b =df
    val n = cross_columns_tuple.size
    val bucket_size=1e5.toInt
    for  (i <- Range(0,n,1)) {
      val c1=cross_columns_tuple(i)(0)
      val c2=cross_columns_tuple(i)(1)
      df_b = cross_column(df_a,c1,c2,bucket_size)
      df_a=df_b

    }
    val cross_df = df_b

    return cross_df

  }



  // val df_new = genearate_new_column(df)
  def genearate_new_column(df:DataFrame,flag:String,n:Int):DataFrame={
    println("starting genearate_new_column ")
    val spark =df.sparkSession

    // 生成新的特征
    val df1 =df.withColumn("user_act_gap",udf( (t:String) => udf_user_act_gap(t) ).apply(col("browse_list_sku_date")))
      .withColumn("flag_browse_sku",udf( (a:String,b:String)=> udf_flag_browse_sku(a,b) ).apply(col("browse_list_sku_date") ,col("sku_id")))
      .withColumn("flag_browse_cate",udf( (a:String,b:String)=> udf_flag_browse_sku(a,b) ).apply(col("browse_list_cate_date") ,col("cate_id")))
      .withColumn("browse_sku_gap",udf( (a:String,b:String)=> udf_browse_sku_gap(a,b) ).apply(col("browse_list_sku_date") ,col("sku_id")))
      .withColumn("browse_cate_gap",udf( (a:String,b:String)=> udf_browse_sku_gap(a,b) ).apply(col("browse_list_cate_date") ,col("cate_id")))
      .withColumn("flag_gen_sku",udf( (a:String,b:String)=> udf_flag_browse_sku(a,b) ).apply(col("gen_list_sku_date") ,col("sku_id")))
      .withColumn("flag_gen_cate",udf( (a:String,b:String)=> udf_flag_browse_sku(a,b) ).apply(col("gen_list_cate_date") ,col("cate_id")))
      .withColumn("gen_sku_gap",udf( (a:String,b:String)=> udf_browse_sku_gap(a,b) ).apply(col("gen_list_sku_date") ,col("sku_id")))
      .withColumn("gen_cate_gap",udf( (a:String,b:String)=> udf_browse_sku_gap(a,b) ).apply(col("gen_list_cate_date") ,col("cate_id")))
//      .withColumn("cross_browse_sku1_sku2",udf( (a:String,b:String)=> udf_cross_sku1_sku2(a,b,0) ).apply(col("browse_list_sku_date") ,col("sku_id")))
//      .withColumn("cross_browse_cate_sku",udf( (a:String,b:String)=> udf_cross_sku1_sku2(a,b,1) ).apply(col("browse_list_cate_date") ,col("cate_id")))
//      .withColumn("cross_gen_sku1_sku2",udf( (a:String,b:String)=> udf_cross_sku1_sku2(a,b,0) ).apply(col("gen_list_sku_date") ,col("sku_id")))
//      .withColumn("cross_gen_cate_sku",udf( (a:String,b:String)=> udf_cross_sku1_sku2(a,b,1) ).apply(col("gen_list_cate_date") ,col("cate_id")))

    //new
//    val df2 = Array2Column(df1,"cross_browse_sku1_sku2",k )
//    val df5 = Array2Column(df2,"cross_browse_cate_sku",k )
//    val df6 = Array2Column(df5,"cross_gen_sku1_sku2",k )
//    val df7 = Array2Column(df6,"cross_gen_cate_sku",k )

    //id
    val df8= thin_column(df1 ,"sku_id",  1e4.toInt , flag ,n)
    val df9= thin_column(df8 ,"brand_id",1e4.toInt , flag ,n)

    // cross
    val df10 = cross_column(df9)

    //return
    val AllColumns = df10.columns
    val useColumns= AllColumns.filter(!(str_columns_del +: "brand_id").contains(_))
    df10.select(useColumns.map(col(_)) : _*)

  }

  // 要删除的string columns，分为
  val str_columns_del = Array( "browse_list_sku_date","search_list_sku_date","fav_list_sku_date","car_list_sku_date","gen_list_sku_date"
    ,"browse_list_cate_date","search_list_cate_date","fav_list_cate_date","car_list_cate_date","gen_list_cate_date"
  )

  val str_columns_direct = Array("sex", "age", "carrer", "marriage", "haschild", "province", "city", "jd_lev","cate_id")
  val str_columns_new = Array("user_act_gap","browse_sku_gap","browse_cate_gap","gen_sku_gap","gen_cate_gap")
  val flag_columns=Array("flag_browse_sku","flag_browse_cate","flag_gen_sku","flag_gen_cate")

  // cate_id看作普通的离散特征处理，
  val id_columns =  Array("sku_id","brand_id")
  val id_columns_thin =  id_columns.map(_+"_thin")
  val id_columns_index =  id_columns_thin.map(_+"_ind")
  val id_columns_vec =  id_columns_index.map(_+"_vec")

  val str_columns_index=(str_columns_direct ++ str_columns_new  ).map(_+"_ind")
  val str_columns_vec=str_columns_index.map(_+"_vec")


  val cross_columns = Array("cross_browse_sku1_sku2","cross_browse_cate_sku","cross_gen_sku1_sku2","cross_gen_cate_sku")
  val cross_columns_buff =ArrayBuffer[String]()

  val k=10
  cross_columns.map{t=>
    for (i <- Range(0,k,1)) {
      cross_columns_buff +=  t+"_"+i
    }
  }
  val cross_columns_s=cross_columns_buff.toArray
  print("cross_columns_s=" + cross_columns_s.mkString(",") )
  val cross_columns_vec=cross_columns_s.map(_+"_vec")

  val useColumns = flag_columns ++ str_columns_vec ++ cross_column_tuple_vec ++ input.id_columns_vec  // ++: cross_columns_vec

  /**
   *
   * @param df :DataFrame
   * @return
   */

  def pip_fit(df:DataFrame ,n:Int): PipelineModel ={
    println("starting input.pip_fit")
    val new_df = genearate_new_column(df,"fit",n)
    new_df.cache()

    val string_index: Array[org.apache.spark.ml.PipelineStage] = (str_columns_direct ++ str_columns_new ++ id_columns_thin ++cross_columns_tuple_s ).map(
      cname => new StringIndexer()
        .setInputCol(cname)
        .setOutputCol(s"${cname}_ind").setHandleInvalid("skip")
    )

    // cross_columns_s非常大，后面容易造成apply数据shuffle出问题。
    val Onehot_Encoder: Array[org.apache.spark.ml.PipelineStage] = (str_columns_index  ++: id_columns_index ++: cross_column_tuple_index ) .map(
      cname => new OneHotEncoder()
        .setInputCol(cname)
        .setOutputCol(s"${cname}_vec")
    )


    //  Assembler
    //  不加入cross_column_vecColumns，加入后，速度非常慢，而且报错
    // xgboost不要使用id_columns_vec和cross_columns_vec，否则会训练速度非常慢。
    val AllColumns = df.schema.fieldNames
    val addColumns = flag_columns ++ str_columns_vec     // ++: cross_columns_vec
    val delColumns = id_columns ++: str_columns_del ++: str_columns_direct  ++: Array("label")
    val VectorInputCols = (AllColumns ++: addColumns).filter(!delColumns.contains(_))


    println("VectorInputCols="+ VectorInputCols.mkString(","))
    val assembler = new VectorAssembler().setInputCols(VectorInputCols).setOutputCol("features")

    val pipeline = new Pipeline().setStages(string_index  ++: Onehot_Encoder ++:Array(assembler) )
    val pModel = pipeline.fit(new_df)
    //save
    val fs = FileSystem.get(new Configuration())
    fs.delete(new Path( s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/pipline" ), true)
    pModel.save(s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/pipline")

    new_df.unpersist()
    return pModel

  }


  def pip_data(df:DataFrame ,n:Int): DataFrame ={
    println("starting input.pip_data")
    val new_df = genearate_new_column(df,"apply",n)
    // load model
    val pModel = PipelineModel.load(s"ads_sz/app.db/app_szad_m_dyrec_rank_model_spark/n=${n}/pipline")
    val transform_df = pModel.transform(new_df)
    return transform_df
  }




}
