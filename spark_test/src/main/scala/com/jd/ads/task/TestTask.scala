package com.jd.ads.task

import org.apache.spark._
import scopt.OptionParser

/**
 * Created by huangzhong on 2016/05/25.
 */
object MixJob extends Logging {// 1. Logging 用于记录用户日志

  // 2. 构建参数，主类的参数通过 --argName指定，如：--inputPath data/input/mix.txt
  case class Params(
                     master: String = "local",
                     inputPath: String = "file:///D://data/input/mix.txt",
                     isMax: Boolean = true,
                     port: Int = 6380)

  def main(args: Array[String]) {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("MixJob") {
      head("MixJob: 做数据处理.")
      opt[String]("master")
        .text(s"master , default: ${defaultParams.master}}")
        .required() // 3. 必须的参数
        .action((x, c) => c.copy(master = x))
      opt[String]("inputPath")
        .text(s"inputPath , default: ${defaultParams.inputPath}}")
        .action((x, c) => c.copy(inputPath = x))
      opt[Boolean]("isMax")
        .text(s"isMax, default: ${defaultParams.isMax}")
        .action((x, c) => c.copy(isMax = x))
      opt[Int]("port")
        .text(s"redis port, default: ${defaultParams.port}")
        .action((x, c) => c.copy(port = x))
      note(
        """
          |For example, the following command runs this app on a mixjob dataset:
          |
          | queue=bdp_jmart_adv.bdp_jmart_sz_ad
          |
          |/software/servers/spark-1.5.2-HADOOP-2.2.0/bin/spark-submit \
          |--master yarn-client \
          |--jars $(echo /home/jd_ad/data_dir/huangzhong/lib/*.jar | tr ' ' ',') \
          |--num-executors 2 \
          |--driver-memory 2g \
          |--executor-memory 2g \
          |--executor-cores 2 \
          |--queue $queue \
          |--conf spark.rdd.compress=true \
          |--conf spark.akka.frameSize=100 \
          |--conf spark.storage.memoryFraction=0.3 \
          |--class com.jd.ads.tools.MixTask /xxx/original-jtfm-1.0.0.jar yarn-client /xx/input/gdt true 6280
          |
        """.stripMargin)
    }

    parser.parse(args, defaultParams).map { params => {
      println("参数：" + params)
      run(params.master,
        params.inputPath,
        params.isMax,
        params.port)
    }
    } getOrElse {
      System.exit(1)
    }
  }

  def run(master: String,
          inputPath: String,
          isMax: Boolean = true,
          port: Int = 6280) = {
    val conf = new SparkConf().setAppName("mixjob").setMaster(master)
    val sc = new SparkContext(conf)
    log.info("do something...")
    sc.stop()
  }
}