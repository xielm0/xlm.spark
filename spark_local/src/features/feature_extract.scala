package features

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by xieliming on 2016/8/26.
 */

object TFIDF {
  def main(args:Array[String]): Unit ={
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}

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
}
/*
[(20,[5,6,9],[0.0,0.6931471805599453,1.3862943611198906]),0]
[(20,[3,5,12,14,18],[1.3862943611198906,0.0,0.28768207245178085,0.28768207245178085,0.28768207245178085]),0]
[(20,[5,12,14,18],[0.0,0.5753641449035617,0.28768207245178085,0.28768207245178085]),1]
*/

//计算词频
object CountVectorizer {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]");
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}

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

/*
  +--------------------+
  |            features|
  +--------------------+
  |(3,[0,1,2],[1.0,1...|
  |(3,[0,1,2],[2.0,2...|
  +--------------------+
  */

