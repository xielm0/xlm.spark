package features

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by xieliming on 2016/8/26.
 */
/*
* VectorSlicer is a transformer that takes a feature vector and outputs a new feature vector with a sub-array of the original features.
* It is useful for extracting features from a vector column.
* */
object rFormula   {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("simple").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import org.apache.spark.ml.feature.RFormula

    val dataset = sqlContext.createDataFrame(Seq(
      (7, "US", 18, 2.0),
      (7, "US", 18, 1.0),
      (8, "CA", 12, 1.0),
      (9, "NZ", 15, 0.0)
    )).toDF("id", "country", "hour", "clicked")
    val formula = new RFormula()
      .setFormula("clicked ~ country + hour")
      .setFeaturesCol("features")
      .setLabelCol("label")
    val output = formula.fit(dataset).transform(dataset)
    output.show()
  }
}
/*
使用公式： clicked ~ country + hour,表明想预测clicked，基于country和hour特征。
so : label = clicked . features : hour是数值，还是hour , ; country是个分类型字符串，也就会转化成数值。按二进制转换。
+---+-------+----+-------+--------------+-----+
| id|country|hour|clicked|      features|label|
+---+-------+----+-------+--------------+-----+
|  7|     US|  18|    2.0|[1.0,0.0,18.0]|  2.0|
|  7|     US|  18|    1.0|[1.0,0.0,18.0]|  1.0|
|  8|     CA|  12|    1.0|[0.0,0.0,12.0]|  1.0|
|  9|     NZ|  15|    0.0|[0.0,1.0,15.0]|  0.0|
+---+-------+----+-------+--------------+-----+
 */
