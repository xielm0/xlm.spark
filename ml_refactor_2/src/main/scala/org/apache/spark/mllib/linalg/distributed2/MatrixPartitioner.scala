package org.apache.spark.mllib.linalg.distributed2


/**
 * Created by xieliming on 2017/2/14.
 */

import org.apache.spark.Partitioner

class MatrixPartitioner (
                          val numRowBlocks: Int,
                          val numColBlocks: Int,
                          val rowsPerPart: Int,
                          val colsPerPart: Int) extends Partitioner {

  require(numRowBlocks > 0)
  require(numColBlocks > 0)
  require(rowsPerPart > 0)
  require(colsPerPart > 0)

    override def numPartitions: Int = numRowBlocks * numColBlocks

    override def getPartition(key: Any): Int = {
      key match {
        case i: Int => i
        case (i: Int, j: Int) =>
          getPartitionId(i, j)
        case (i: Int, j: Int, _: Int) =>
          getPartitionId(i, j)
        case _ =>
          throw new IllegalArgumentException(s"Unrecognized key: $key.")
      }
    }

  /** Partitions sub-matrices as blocks with neighboring sub-matrices. */
  private def getPartitionId(i: Int, j: Int): Int = {
    require(0 <= i && i < numRowBlocks, s"Row index $i out of range [0, $numRowBlocks).")
    require(0 <= j && j < numColBlocks, s"Column index $j out of range [0, $numColBlocks).")
    i + j  * numRowBlocks
  }

  override def equals(obj: Any): Boolean = {
    obj match {
      case r: MatrixPartitioner =>
        (this.numRowBlocks == r.numRowBlocks) && (this.numColBlocks == r.numColBlocks) &&
          (this.rowsPerPart == r.rowsPerPart) && (this.colsPerPart == r.colsPerPart)
      case _ =>
        false
    }
  }
}
