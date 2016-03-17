package com.jd.spark.ml

import scala.util.Random

 /**
  * get one sample from some distribution
  */

 //Multinomial 多项式
 //将cnt正则化后，加和，最后返回中间任意1个值的索引值
object Sampler
{
  def Multinomial(arrInput: Array[Double]): Int = {
    val rand = Random.nextDouble()
    val s = doubleArrayOps(arrInput).sum  //等同 arrInput.sum
    val arrNormalized = doubleArrayOps(arrInput).map { e => e / s }
    var cum = 0.0
    val cumArr = doubleArrayOps(arrNormalized).map {
      t =>
      cum = cum + t
      cum
    }
    doubleArrayOps(cumArr).indexWhere(cumDist => cumDist >= rand)
  }

  def Uniform(dimension: Int,
              rand: Random): Int =
  { 
    rand.nextInt(dimension)
  }
}
