package com.jd.szad.userlabel

import com.jd.szad.tools.Writer
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.sql.functions._
import org.apache.spark.{SparkConf, SparkContext}
/**
 * Created by xieliming on 2016/10/31.
 * 基于用户标签的推荐算法
 * 用户 x label , label x sku 的score用提升度
 * spark-shell --num-executors 10 --executor-memory 8g --executor-cores 4
 */
object app {

  def main(args:Array[String]) {
    val conf = new SparkConf()
      .setAppName("user_label")
      .set("spark.akka.timeout", "1000")
      .set("spark.rpc.askTimeout", "500")
      .set("spark.storage.memoryFraction","0.1")

    val sc = new SparkContext(conf)

    //params
    val model_type =args(0)

    val sqlContext = new org.apache.spark.sql.hive.HiveContext(sc)

    if (model_type =="train") {
//      sqlContext.sql("set spark.sql.shuffle.partitions = 400")
      val sql =
        """
          |select  uid,type,label,sku
          |from app.app_szad_m_dyrec_userlabel_train
          |where type in('209001','210001','211001','309001','310001','311001','331001','332001','333001','363001','364001','365001')
        """.stripMargin
      val org_df = sqlContext.sql(sql)
      val df = org_df.rollup("type", "label","sku").agg(countDistinct("uid") as "uv").filter(col("uv")>30).cache()

      // df.filter(df("type")===1)

      // type,label,sku
      val df1 = df.filter("sku is not null").selectExpr("type","label","sku","uv as sku_uv")
      val df2 = df.filter("sku is null and label is not null ").selectExpr("type","label","uv as label_uv")
      val df3 = df.filter("sku is null and label is null and type is not null").selectExpr("type","uv as type_uv")
      val df4 = df1.groupBy("type","sku").agg(sum("sku_uv") as "sku_uv" )
      println("df3.count is " + df3.count())

      // prob
      val sku_label_rate = df1.join(df2, Seq("type","label")).selectExpr("type","label", "sku", "round(sku_uv/label_uv,8) as sku_label_rate")
      val sku_type_rate = df4.join(df3, Seq("type")).selectExpr("type", "sku", "round(sku_uv/type_uv,8) as sku_type_rate")
      // lift
      val lift = sku_label_rate.join(sku_type_rate,Seq("type", "sku")).selectExpr("type", "label", "sku", "round(sku_label_rate/sku_type_rate,4) as lift")
      val lift_1 = lift.where(col("lift")>1)

      // require boost_prob>1,so log(2,boost_prob)>0
      // 性别、年龄的提升度不会很大，但是对于类目这种，则value is large
      //  test
      //  df.where(col("label")===9729 && col("sku")==="10613996786").show()

      //save
      //boost_prob.write.save("app.db/app_szad_m_dyRec_userlabel_model")
      sqlContext.sql("use app")
      lift_1.registerTempTable("res_table")
      sqlContext.sql("set mapreduce.output.fileoutputformat.compress=true")
      sqlContext.sql("set hive.exec.compress.output=true")
      sqlContext.sql("set mapred.output.compression.codec=com.hadoop.compression.lzo.LzopCodec")
      sqlContext.sql("insert overwrite table app.app_szad_m_dyrec_userlabel_model select type,label,sku,lift from res_table")

     }else if (model_type =="sql_predict") {
      /* 用户 x label , label x sku ,这里，一个用户不超过100个标签，一个label对应不超过100个sku。否则计算困难。
       * 最终保存的结果，每个用户保存top 100
       *  */
      sqlContext.sql("set spark.sql.shuffle.partitions = 1000")

      val sql=
        """
          |insert overwrite table app.app_szad_m_dyRec_userlabel_predict_res partition (user_type=1)
          |select uid ,sku ,score,rn
          |  from( select uid ,sku ,score ,row_number() over(partition by uid order by score desc) rn
          |          from(select uid,sku,sum(round(rate*score,1)) as score
          |                from (select uid,type,label,rate
          |                      from(select uid,type,label,rate,row_number() over(partition by uid order by type,label) rn
          |                            from(select gdt_openid as uid,label_code as type,label_value as label,1 as rate,row_number() over(partition by gdt_openid order by label_code,label_value) rn
          |                                  from app.app_szad_m_dmp_label_gdt_openid
          |                                 where label_type in('209','210','211','309','310','311','331','332','333','363','364','365')
          |                                   and label_code in('209001','210001','211001','309001','310001','311001','331001','332001','333001','363001','364001','365001')
          |                                  )a1
          |                           )a2
          |                      where rn <100 )a
          |                join (select sku,type,label,score
          |                       from (select sku,type,label,score,row_number() over(partition by type,label order by score desc ) rn
          |                               from app.app_szad_m_dyRec_userlabel_model)t
          |                       where rn <=100 )b
          |                 on (a.type=b.type and a.label=b.label)
          |              group by uid,sku
          |                )t1
          |       )t2
          | where rn <=100
        """.stripMargin
      sqlContext.sql(sql)

    }else if (model_type =="predict") {

      val sql=
        """
          |select uid ,concat(type,'_',label) as label,1 as rate
          |from app.app_szad_m_midpage_jd_label_train
          |where length(uid)<=40
          |group by uid,concat(type,'_',label)
        """.stripMargin
      val user_label = sqlContext.sql(sql).rdd.map(t=>(t(0),t(1),t(2).asInstanceOf[Int]))

      val sql2 =
        """
          |select label,sku,score
          |from(
          |select concat(type,'_',label) as label,sku,score,row_number() over(partition by type,label order by score desc ) rn
          |from app.app_szad_m_midpage_jd_label_model
          |)t where rn<=200
        """.stripMargin
      val label_sku =sqlContext.sql(sql2).rdd.map(t=>(t(0),t(1),t(2).asInstanceOf[Double])).cache()

      //labels map
      val labels = sc.broadcast(label_sku.map(_._1).distinct().zipWithIndex().collectAsMap() )
      val skus   = sc.broadcast(label_sku.map(_._2).distinct().zipWithIndex().collectAsMap() )
      val users  = user_label.map(_._1).distinct().zipWithIndex()

      //
      val A = user_label.map(t=>(t._1,(t._2,t._3))).join(users).
        map{ t =>
          val user = t._2._2
          if (labels.value.contains(t._2._1._1) ){
              val label = labels.value( t._2._1._1)
              MatrixEntry(user,label,t._2._1._2)
          }else{MatrixEntry(-1,-1,0)}
        }.filter(t=>t.i>=0)

      val B = label_sku.map{ t =>
        val label = labels.value( t._1)
        val sku = skus.value( t._2)
        MatrixEntry(label,sku,t._3)
      }

      //compute
      // block matrix *  block matrix
      val mat_A  = new CoordinateMatrix( A ).toBlockMatrix(1024,1024)
      val mat_B  = new CoordinateMatrix( B ).toBlockMatrix(1024,1024)
      val S = mat_A.multiply(mat_B)

      println("numRowBlocks= " + mat_A.numRowBlocks + ",numColBlocks"+mat_B.numColBlocks)  //numColBlocks=180
      println("A.blocks.partitions.length"+mat_A.blocks.partitions.length + "B.blocks.partitions.length"+mat_B.blocks.partitions.length)
      println("first block size = " + S.blocks.map(t=>t._2.toString()))

      val res =S.toCoordinateMatrix().entries.map(t=>t.i + "\t" + t.j + "\t" + t.value)

      //save
      Writer.write_table( res ,"app.db/app_szad_m_midpage_jd_label_res","lzo")

    }
  }


}
