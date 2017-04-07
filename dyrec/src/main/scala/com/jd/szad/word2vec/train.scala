package com.jd.szad.word2vec

import com.jd.szad.tools.Writer
import org.apache.spark.mllib.feature.Word2Vec
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by xieliming on 2017/4/5.
 */
object train {
  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setAppName("OCPA.word2vec")
      .set("spark.akka.timeout", "1000")
      .set("spark.rpc.askTimeout", "500")

    val sc = new SparkContext(conf)
    val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)
    val dt:String = args(0)

    val s0=
      s"""
        |select uid,sku,rn
        |from app.app_szad_m_dyrec_user_top100_sku
        |where user_type=0 and action_type=1
        |and dt='${dt}' and date='${dt}'
      """.stripMargin
    val input :RDD[Seq[String]]= hiveContext.sql(s0).map(t=>
      (t.getAs("uid").toString,(t.getAs("sku").toString,t.getAs("rn").asInstanceOf[Int]))
    ).groupByKey().map{
      case( key, iter)=>  //iter=Iterable[(String,Int)]
        iter.toArray.sortBy(_._2).reverse.map(_._1).toSeq
    }

    val minCount=20
    val word2vec = new Word2Vec().setVectorSize(100).setLearningRate(0.01).setMinCount(minCount).setNumPartitions(100)

    val model = word2vec.fit(input)

    val s1=
      s"""
        |select sku
        |from(select sku,count(1) as cnt
        |      from app.app_szad_m_dyrec_user_top100_sku
        |      where user_type=0 and action_type=1
        |      and dt='${dt}' and date='${dt}'
        |      group by sku
        |    )t
        |where cnt >= ${minCount}
      """.stripMargin
    val vob =hiveContext.sql(s1).map(t=> t.getAs("sku").toString)
//    print("vob's count =" + vob.count())

    val res= vob.repartition(1000).map { sku =>
      val synonyms = model.findSynonyms(sku, 10).map(t=>t._1).mkString(",")
      val s= sku + "\t" + synonyms
      s
    }

    //save
    Writer.write_table(res,"app.db/ads_sz/app_szad_m_dyrec_word2vec_model_res","lzo")


//    val synonyms = model.findSynonyms("1", 5)
//    for((synonym, cosineSimilarity) <- synonyms) {
//      println(s"$synonym $cosineSimilarity")
//    }

    // Save and load model
//    model.save(sc, "myModelPath")
//    val sameModel = Word2VecModel.load(sc, "myModelPath")


  }
}
