package tools

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.rdd.RDD

/**
 * Created by xieliming on 2016/6/7.
 */
object Writer {
  def write_table(rdd_context:RDD[String] , hdfs_path :String ): Unit ={
    //删除mode文件
    val fsconf = new Configuration()
    val fs = FileSystem.get(fsconf)
    fs.delete(new Path( hdfs_path ),true)
    //保存到数据库
    rdd_context.saveAsTextFile(hdfs_path )
  }

  def write_table(rdd_context:RDD[String] , hdfs_path :String ,codec :String): Unit ={
    //删除mode文件
    val fsconf = new Configuration()
    val fs = FileSystem.get(fsconf)
    fs.delete(new Path( hdfs_path ),true)
    //保存到数据库
    if (codec=="lzo") {
      rdd_context.saveAsTextFile(hdfs_path, classOf[LzopCodec])
    }
  }

  //model.userFeatures.toDF("id", "features").write.parquet(userPath(path))

}
