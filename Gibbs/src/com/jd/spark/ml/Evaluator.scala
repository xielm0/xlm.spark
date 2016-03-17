package com.jd.spark.ml

import org.apache.spark.Logging
import org.apache.spark.rdd.RDD

/**
   * Perplexity is a kind of evaluation method of LDA. Usually it is used on unseen data. But here
   * we use it for current documents, which is also OK. If using it on unseen data, you must do an
   * iteration of Gibbs sampling before calling this. Small perplexity means good result.
   */

object Evaluator extends Logging
{
  def perplexity( data: RDD[Document], 
                  phi: Array[Array[Double]], 
                  theta: Array[Array[Double]]): Double = {
    val (termProb, totalNum) = data.flatMap 
    {
      case Document(docId, content) =>

      val vSize = phi.head.length
      val currentTheta = new Array[Double](vSize)
      var col = 0
      var row = 0
      while (col < phi.head.size) {
        row = 0
        while (row < phi.length) {
          currentTheta(col) += phi(row)(col) * theta(docId.toInt)(row)
          row += 1
        }
        col += 1
      }
      content.map(x => (math.log(currentTheta(x)), 1))
    }.reduce { (lhs, rhs) =>
      (lhs._1 + rhs._1, lhs._2 + rhs._2)
    }
    math.exp(-1 * termProb / totalNum)
  }
}
