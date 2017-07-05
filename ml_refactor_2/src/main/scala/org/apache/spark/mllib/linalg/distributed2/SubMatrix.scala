package org.apache.spark.mllib.linalg.distributed2

import breeze.linalg.{DenseMatrix => BDM}
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.SparseMatrix

class SubMatrix() extends Serializable with Logging {

  private var denseBlock: BDM[Double] = null

  private var sparseBlock: SparseMatrix = null

  protected var sparse: Boolean = true

  protected var nonZeros: Long = 0L

  def this(denseMatrix: BDM[Double]) = {
    this()
    sparse = false
    denseBlock = denseMatrix
  }

  def this(spMatrix: SparseMatrix) = {
    this()
    sparseBlock = spMatrix
  }

  def rows: Int = {
    if (sparse) {
      sparseBlock.numRows
    } else denseBlock.rows
  }

  def cols: Int = {
    if (sparse) {
      sparseBlock.numCols
    } else denseBlock.cols
  }

  def isSparse = this.sparse

  def add(other: SubMatrix): SubMatrix = {
    (this.isSparse, other.isSparse) match {
      case (false, false) =>
        val res: BDM[Double] = this.denseBlock + other.denseBlock
        new SubMatrix(denseMatrix = res)
      case _ =>
        throw new IllegalArgumentException(s"Not supported add-operator between matrices of sparsity " +
          s"with ${this.isSparse} and ${other.isSparse}")
    }
  }


//  def multiply(other: SubMatrix): SubMatrix = {
//    (this.isSparse, other.isSparse) match {
//      case (true, true) =>
//        val c: BDM[Double] = this.sparseBlock.multiply(other.sparseBlock)
//        new SubMatrix(denseMatrix = c)
//      case _ =>
//        throw new IllegalArgumentException(s"Not supported multiply-operator between matrices of sparsity " +
//          s"with ${this.isSparse} and ${other.isSparse}")
//    }
//  }
}



