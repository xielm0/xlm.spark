package tools

/**
 * Created by xieliming on 2016/6/15.
 */

import org.apache.spark.mllib.linalg.BLAS.dot
import org.apache.spark.mllib.linalg.{Vector, Vectors}

object dist {
  /* Compute the cosine similarity between two vectors */
  def cosine(vec1: Array[Double], vec2: Array[Double]): Double = {
    cosine(Vectors.dense(vec1),Vectors.dense(vec2))
  }

  def cosine(vec1 : Vector , vec2 : Vector) :Double={
    dot(vec1,vec2) / (Vectors.norm(vec1,2) * Vectors.norm(vec2,2))
  }
}
