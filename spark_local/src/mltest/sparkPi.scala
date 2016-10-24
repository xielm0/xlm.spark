package mltest

import org.apache.spark.{SparkConf, SparkContext}
import scala.math._

/**
 * Created by xieliming on 2016/8/25.
 */
object sparkPi {
  def main(args: Array[String]) {

    val conf = new SparkConf().setMaster("local[*]").setAppName("SparkPi")
    //创建环境变量 设置本地模式 设置所要执行APP的名字
    val sc = new SparkContext(conf)
    //创建环境变量实例
    val slices = if (args.length > 0) args(0).toInt else 2
    val n = math.min(100000L * slices, Int.MaxValue).toInt
    //随机产生100000个数
    val count = sc.parallelize(1 until n, slices).map { i =>
      val x = random * 2 - 1
      val y = random * 2 - 1
      if (x * x + y * y < 1) 1 else 0
    }.reduce(_ + _)
    println("Pi is rough：" + 4.0 * count / n)
    sc.stop()
  }
}
