package com.jd.szad.pb

import org.apache.spark.sql.SparkSession


/**
 * Created by xieliming on 2017/4/27.
 */
object PersonParser {
  def serialize: Array[Byte] =  {
    val builder = PersonProto.Person.newBuilder()

    builder.setName("xieliming")
    builder.setId(1)
    builder.setEmail("abd@jd.com")
    builder.addPhone("137")

    builder.build().toByteArray
  }

  def deserialize(x: Array[Byte]) {
    val parser = PersonProto.Person.parseFrom(x)
    val name = parser.getName
    val id = parser.getId
    (name,id)
  }

  /*pb的保存是个难题。
  * rdd保存为txt，是以\n分割，但pb是一个二进制流（Array[Byte]）,不能以\n分割
  *
  * */
  def main: Unit ={
    val spark = SparkSession.builder()
      .appName("pb.test")
      .enableHiveSupport()
      .getOrCreate()

    val test=serialize
    val res=Array(test)

    val rdd1=spark.sparkContext.parallelize(res)
    rdd1.saveAsTextFile("")

  }




}
