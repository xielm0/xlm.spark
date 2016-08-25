package dm_train

/**
 * Created by xieliming on 2016/7/14.
 */

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SQLContext
object NN {
  def main(args:Array[String]) {

    val sparkConf = new SparkConf().setAppName("LR")
    val sc = new SparkContext(sparkConf)

    // Load training data
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_multiclass_classification_data.txt").toDF()
    // Split the data into train and test
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)
    // specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4 and output of size 3 (classes)
    val layers = Array[Int](4, 5, 4, 3)
    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)
    // train the model
    val model = trainer.fit(train)
    // compute precision on the test set
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("precision")
    println("Precision:" + evaluator.evaluate(predictionAndLabels))
  }
}
