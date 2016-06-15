package com.jd.szad.als

/**
 * Created by xieliming on 2016/6/15.
 */

import breeze.linalg.no
import org.apache.spark.mllib.linalg.{Vectors, BLAS}

//import org.jblas.DoubleMatrix

object dist {
  /* Compute the cosine similarity between two vectors */
  def cosine(vec1: Array[Double], vec2: Array[Double]): Double = {
    //vec1.dot(vec2) / (vec1.norm2() * vec2.norm2())
    val a = Vectors.dense(vec1)
    val b = Vectors.dense(vec2)
    BLAS.dot(a,b)/(norm(a) * normL2(b))
  }
}
