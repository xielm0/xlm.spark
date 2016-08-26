package features

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by xieliming on 2016/8/25.
 */
  //分词，类似split，返回结果类型不一样
/*Tokenization is the process of taking text (such as a sentence) and breaking it into individual terms (usually words).
A simple Tokenizer class provides this functionality. The example below shows how to split sentences into sequences of words.
*/
object token {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("simple").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}

    val sentenceDataFrame = sqlContext.createDataFrame(Seq(
      (0, "Hi I heard about Spark"),
      (1, "I wish Java could use case classes"),
      (2, "Logistic,regression,models,are,neat")
    )).toDF("label", "sentence")
    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val tokenized = tokenizer.transform(sentenceDataFrame)
    tokenized.select("words", "label").take(3).foreach(println)

    val regexTokenizer = new RegexTokenizer().setInputCol("sentence").setOutputCol("words")
      .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)
    val regexTokenized = regexTokenizer.transform(sentenceDataFrame)
    regexTokenized.select("words", "label").take(3).foreach(println)
  }
}
  /*
  [WrappedArray(hi, i, heard, about, spark),0]
  [WrappedArray(i, wish, java, could, use, case, classes),1]
  [WrappedArray(logistic,regression,models,are,neat),2]

  [WrappedArray(Hi, I, heard, about, Spark),0]
  [WrappedArray(I, wish, Java, could, use, case, classes),1]
  [WrappedArray(Logistic, regression, models, are, neat),2]
  */

//分词，得到词组合
/*An n-gram is a sequence of nn tokens (typically words) for some integer nn. The NGram class can be used to transform input features into nn-grams.

NGram takes as input a sequence of strings (e.g. the output of a Tokenizer). The parameter n is used to determine the number of terms in each nn-gram.
The output will consist of a sequence of nn-grams where each nn-gram is represented by a space-delimited string of nn consecutive words.
If the input sequence contains fewer than n strings, no output is produced.*/
object n_gram {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("simple").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import org.apache.spark.ml.feature.NGram

    val wordDataFrame = sqlContext.createDataFrame(Seq(
      (0, Array("Hi", "I", "heard", "about", "Spark")),
      (1, Array("I", "wish", "Java", "could", "use", "case", "classes")),
      (2, Array("Logistic", "regression", "models", "are", "neat"))
    )).toDF("label", "words")

    val ngram = new NGram().setInputCol("words").setOutputCol("ngrams")
    val ngramDataFrame = ngram.transform(wordDataFrame)
    ngramDataFrame.take(3).map(_.getAs[Stream[String]]("ngrams").toList).foreach(println)
  }
}
/*
List(Hi I, I heard, heard about, about Spark)

List(I wish, wish Java, Java could, could use, use case, case classes)
List(Logistic regression, regression models, models are, are neat)
*/

//转化成0/1 .根据setThreshold的设置
object binary {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("simple").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import org.apache.spark.ml.feature.Binarizer
    import org.apache.spark.sql.DataFrame

    val data = Array(
      (0, 0.1),
      (1, 0.8),
      (2, 0.2)
    )
    val dataFrame: DataFrame = sqlContext.createDataFrame(data).toDF("label", "feature")

    val binarizer: Binarizer = new Binarizer()
      .setInputCol("feature")
      .setOutputCol("binarized_feature")
      .setThreshold(0.5)

    val binarizedDataFrame = binarizer.transform(dataFrame)  //按照binarizer规则去转换
    val binarizedFeatures = binarizedDataFrame.select("binarized_feature")
    binarizedFeatures.collect().foreach(println)
  }
}
/*
[0.0]
[1.0]
[0.0]
*/

//分桶，也是按照阈值，只是阈值是一个Array .如：Array(Double.NegativeInfinity, -0.5, 0.0, 0.5, Double.PositiveInfinity)
object bucket {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("simple").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import org.apache.spark.ml.feature.Bucketizer

    val splits = Array(Double.NegativeInfinity, -0.5, 0.0, 0.5, Double.PositiveInfinity)

    val data = Array(-0.5, -0.3, 0.0, 0.2)
    val dataFrame = sqlContext.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val bucketizer = new Bucketizer()
      .setInputCol("features")
      .setOutputCol("bucketedFeatures")
      .setSplits(splits)

    // Transform original data into its bucket index.
    val bucketedData = bucketizer.transform(dataFrame)
  }
}

//元素点积
object Element_wise_Product  {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("simple").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import org.apache.spark.ml.feature.ElementwiseProduct
    import org.apache.spark.mllib.linalg.Vectors

    // Create some vector data; also works for sparse vectors
    val dataFrame = sqlContext.createDataFrame(Seq(
      ("a", Vectors.dense(1.0, 2.0, 3.0)),
      ("b", Vectors.dense(4.0, 5.0, 6.0)))).toDF("id", "vector")

    val transformingVector = Vectors.dense(0.0, 1.0, 2.0)
    val transformer = new ElementwiseProduct()
      .setScalingVec(transformingVector)
      .setInputCol("vector")
      .setOutputCol("transformedVector")

    // Batch transform the vectors to create new column:
    transformer.transform(dataFrame).show()
  }
}
/*
+---+-------------+-----------------+
| id|       vector|transformedVector|
+---+-------------+-----------------+
|  a|[1.0,2.0,3.0]|    [0.0,2.0,6.0]|
|  b|[4.0,5.0,6.0]|   [0.0,5.0,12.0]|
+---+-------------+-----------------+
*/

//正太分布正则化
object norm  {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("simple").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import org.apache.spark.ml.feature.Normalizer
    import org.apache.spark.mllib.util.MLUtils

    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
    val dataFrame = sqlContext.createDataFrame(data)

    // Normalize each Vector using $L^1$ norm.
    val normalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(1.0)
    val l1NormData = normalizer.transform(dataFrame)

    // Normalize each Vector using $L^\infty$ norm.
    val lInfNormData = normalizer.transform(dataFrame, normalizer.p -> Double.PositiveInfinity)
  }
}

//将特征的字段合并
/*
Assume that we have a DataFrame with the columns id, hour, mobile, userFeatures, and clicked:
id | hour | mobile | userFeatures     | clicked
----|------|--------|------------------|---------
0  | 18   | 1.0    | [0.0, 10.0, 0.5] | 1.0
userFeatures is a vector column that contains three user features. We want to combine hour, mobile,
and userFeatures into a single feature vector called features and use it to predict clicked or not.
If we set VectorAssembler’s input columns to hour, mobile, and userFeatures and output column to features,
 after transformation we should get the following DataFrame:

id | hour | mobile | userFeatures     | clicked | features
----|------|--------|------------------|---------|-----------------------------
0  | 18   | 1.0    | [0.0, 10.0, 0.5] | 1.0     | [18.0, 1.0, 0.0, 10.0, 0.5]
*/
object vector_assembler  {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("simple").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import org.apache.spark.mllib.linalg.Vectors
    import org.apache.spark.ml.feature.VectorAssembler

    val dataset = sqlContext.createDataFrame(
      Seq((0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0))
    ).toDF("id", "hour", "mobile", "userFeatures", "clicked")
    dataset.show()
    val assembler = new VectorAssembler()
      .setInputCols(Array("hour", "mobile", "userFeatures"))
      .setOutputCol("features")
    val output = assembler.transform(dataset)
    output.show()
  }
}

//多项式扩展特征多项式空间
/*Polynomial expansion is the process of expanding your features into a polynomial space,
which is formulated by an n-degree combination of original dimensions.

例如：[-2.0,2.3] 扩展到3项式空间是
[[-2.0,4.0,-8.0,2.3,-4.6,9.2,5.289999999999999,-10.579999999999998,12.166999999999996]] 即： （-2）,(-2)^2,(-2)^3
*/
object polynomial_expansion  {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("simple").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import org.apache.spark.ml.feature.PolynomialExpansion
    import org.apache.spark.mllib.linalg.Vectors

    val data = Array(
      Vectors.dense(-2.0, 2.3),
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.6, -1.1)
    )
    val df = sqlContext.createDataFrame(data.map(Tuple1.apply)).toDF("features")
    val polynomialExpansion = new PolynomialExpansion()
      .setInputCol("features")
      .setOutputCol("polyFeatures")
      .setDegree(3)  //扩展到3次多项式空间
    val polyDF = polynomialExpansion.transform(df)
    //polyDF.show()
    polyDF.select("polyFeatures").take(3).foreach(println)
  }
}

/*
[[-2.0,4.0,-8.0,2.3,-4.6,9.2,5.289999999999999,-10.579999999999998,12.166999999999996]] 即： （-2）,(-2)^2,(-2)^3
[[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]
[[0.6,0.36,0.216,-1.1,-0.66,-0.396,1.2100000000000002,0.7260000000000001,-1.3310000000000004]]
*/

//Discrete Cosine Transfor 离散余弦变换
//它类似于离散傅里叶变换DFT , 但是只使用实数.经常被信号处理和图像处理使用.
object DCT  {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("simple").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import org.apache.spark.ml.feature.DCT
    import org.apache.spark.mllib.linalg.Vectors

    val data = Seq(
      Vectors.dense(0.0, 1.0, -2.0, 3.0),
      Vectors.dense(-1.0, 2.0, 4.0, -7.0),
      Vectors.dense(14.0, -2.0, -5.0, 1.0))
    val df = sqlContext.createDataFrame(data.map(Tuple1.apply)).toDF("features")
    val dct = new DCT()
      .setInputCol("features")
      .setOutputCol("featuresDCT")
      .setInverse(false)
    val dctDf = dct.transform(df)
    dctDf.select("featuresDCT").take(3).foreach(println)
  }
}

/*
[[1.0,-1.1480502970952693,2.0000000000000004,-2.7716385975338604]]
[[-1.0,3.378492794482933,-7.000000000000001,2.9301512653149677]]
[[4.0,9.304453421915744,11.000000000000002,1.5579302036357163]]
*/
