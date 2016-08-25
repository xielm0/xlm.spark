package features

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by xieliming on 2016/8/25.
 */
object simple {
   def main(args:Array[String]): Unit ={
     val conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]");
     val sc = new SparkContext(conf)

     //test
//     TFIDF(sc)
     CountVectorizer(sc)

   }

  def TFIDF(sc:SparkContext): Unit ={
    import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}

    val sqlContext = new SQLContext(sc)
    //TF-IDF
    val sentenceData = sqlContext.createDataFrame(Seq(
      (0, "Hi I heard about Spark"),
      (0, "I wish Java could use case classes"),
      (1, "Logistic regression models are neat")
    )).toDF("label", "sentence")
    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)
    val featurizedData = hashingTF.transform(wordsData)
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("features", "label").take(3).foreach(println)
  }

  def CountVectorizer(sc:SparkContext): Unit ={
    import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}

    val sqlContext = new SQLContext(sc)
    val df = sqlContext.createDataFrame(Seq(
      (0, Array("a", "b", "c")),
      (1, Array("a", "b", "b", "c", "a"))
    )).toDF("id", "words")

    // fit a CountVectorizerModel from the corpus
    val cvModel: CountVectorizerModel = new CountVectorizer()
  .setInputCol("words")
  .setOutputCol("features")
  .setVocabSize(3)
  .setMinDF(2) // a term must appear in more or equal to 2 documents to be included in the vocabulary
  .fit(df)

    // alternatively, define CountVectorizerModel with a-priori vocabulary
    val cvm = new CountVectorizerModel(Array("a", "b", "c"))
      .setInputCol("words")
      .setOutputCol("features")

    cvModel.transform(df).select("features").show()
  }
 }
