package com.jd.spark.ml

import scala.util.Random

 /**
  * get one sample from some distribution
  */

object Sampler
{
  def Multinomial(arrInput: Array[Double]): Int = {
    val rand = Random.nextDouble()
    val s = doubleArrayOps(arrInput).sum
    val arrNormalized = doubleArrayOps(arrInput).map { e => e / s }
    var localSum = 0.0
    val cumArr = doubleArrayOps(arrNormalized).map 
    { 
      dist =>
        
      localSum = localSum + dist
      localSum
    }
    doubleArrayOps(cumArr).indexWhere(cumDist => cumDist >= rand)
  }

  def Uniform(dimension: Int,
              rand: Random): Int =
  { 
    rand.nextInt(dimension)
  }
}
