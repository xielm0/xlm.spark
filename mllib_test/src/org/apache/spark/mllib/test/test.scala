package org.apache.spark.mllib.test

/**
 * Created by xieliming on 2015/12/22.
 */
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by xieliming on 2015/11/12.
 */
//spark-submit --master yarn-client --class org.apache.spark.mllib.test.test /home/jd_ad/xlm/mllib_test.jar
object test {
  def main(args:Array[String]) {
    val sparkConf = new SparkConf().setAppName("mllib_test")
    val sc = new SparkContext(sparkConf)

    val data = sc.textFile("app.db/xlm_test_mllib/sample_naive_bayes_data.txt") //return array[string]

    val parseData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }

    val training=parseData

    val aggregated = training.map(p => (p.label, p.features)).combineByKey[(Long, Vector)](
       createCombiner =(v: Vector) => {
       val b=v.copy.toArray
        (1L, Vectors.dense(b))
      },
      mergeValue =(c: (Long, Vector), v: Vector) => {
        BLAS.axpy(1.0, v, c._2)
        (c._1 + 1L, c._2)
      },
      mergeCombiners = (c1: (Long, Vector), c2: (Long, Vector)) => {
        BLAS.axpy(1.0, c2._2, c1._2)
        (c1._1 + c2._1, c1._2)
      }
    ).collect()

    //打印
    aggregated.foreach { //case (_, (n, _)) =>
      t=>
        println("label:",t._1 ,"|count:",t._2._1,"|value:",t._2._2.toArray.mkString(","))
    }
   /*结果：
(label:,2.0,|count:,4,|value:,0.0,0.0,10.0)
(label:,0.0,|count:,4,|value:,10.0,0.0,0.0)
(label:,1.0,|count:,4,|value:,0.0,10.0,0.0)
    */
  }
}
