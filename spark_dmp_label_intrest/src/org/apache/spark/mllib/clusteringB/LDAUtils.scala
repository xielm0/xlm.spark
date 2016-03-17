package org.apache.spark.mllib.clusteringB

/**
 * Created by xieliming on 2015/11/10.
 */
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, max, sum}
import breeze.numerics._

/**
 * Utility methods for LDA.
 */
private[clusteringB] object LDAUtils {
  /**
   * Log Sum Exp with overflow protection using the identity:
   * For any a: \log \sum_{n=1}^N \exp\{x_n\} = a + \log \sum_{n=1}^N \exp\{x_n - a\}
   */
  private[clusteringB] def logSumExp(x: BDV[Double]): Double = {
    val a = max(x)
    a + log(sum(exp(x :- a)))
  }

  /**
   * For theta ~ Dir(alpha), computes E[log(theta)] given alpha. Currently the implementation
   * uses [[breeze.numerics.digamma]] which is accurate but expensive.
   */
  private[clusteringB] def dirichletExpectation(alpha: BDV[Double]): BDV[Double] = {
    digamma(alpha) - digamma(sum(alpha))
  }

  /**
   * Computes [[dirichletExpectation()]] row-wise, assuming each row of alpha are
   * Dirichlet parameters.
   */
  private[clusteringB] def dirichletExpectation(alpha: BDM[Double]): BDM[Double] = {
    val rowSum = sum(alpha(breeze.linalg.*, ::))
    val digAlpha = digamma(alpha)
    val digRowSum = digamma(rowSum)
    val result = digAlpha(::, breeze.linalg.*) - digRowSum
    result
  }

}
