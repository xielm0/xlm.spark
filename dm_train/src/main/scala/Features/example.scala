package Features

/**
 * Created by xieliming on 2016/8/24.
 */

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

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
//缺点是自己要给出阈值。
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

//将数值型转换到一个二进制向量列，常用于逻辑回归。
//图中的例子，是先把“a”“b”“c”转换成数值，再将数值转换为二进制向量。
object one_hot_encoding  {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("simple").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}

    val df = sqlContext.createDataFrame(Seq(
      (0, "a"),
      (1, "b"),
      (2, "c"),
      (3, "a"),
      (4, "a"),
      (5, "c")
    )).toDF("id", "category")

    val indexer = new StringIndexer()  //给index编号
      .setInputCol("category")
      .setOutputCol("categoryIndex")
      .fit(df)
    val indexed = indexer.transform(df)

    val encoder = new OneHotEncoder().setInputCol("categoryIndex").
      setOutputCol("categoryVec")
    val encoded = encoder.transform(indexed)
    encoded.show()
  }
}
/*
| id|category|categoryIndex|  categoryVec|
+---+--------+-------------+-------------+
|  0|       a|          0.0|(2,[0],[1.0])|
|  1|       b|          2.0|    (2,[],[])|
|  2|       c|          1.0|(2,[1],[1.0])|
|  3|       a|          0.0|(2,[0],[1.0])|
|  4|       a|          0.0|(2,[0],[1.0])|
|  5|       c|          1.0|(2,[1],[1.0])|
*/

