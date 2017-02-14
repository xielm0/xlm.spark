package org.apache.spark.mllib.linalg.distributed2

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

class SparseVecMatrix(
    val rows: RDD[(Long, BSV[Double])],
    val nRows: Long,
    val nCols: Long) {

  /**
   * multiply another matrix in SparseVecMatrix type
   *
   * @param other another matrix to be multiplied in SparseVecMatrix type
   */
  final def multiplySparse(other: SparseVecMatrix): CoordinateMatrix = {
    val thisEmits = rows.flatMap{case(index, values) =>
      val arr = new ArrayBuffer[(Long, (Long, Float))]
      val len = values.index.size
      for (i <- 0 until len){
        arr += ((values.index(i), (index, values.data(i).toFloat)))
      }
      arr
    }.groupByKey()

    val otherEmits = other.rows.flatMap{case(index, values) =>
      val arr = new ArrayBuffer[(Long, (Long, Float))]()
      val len = values.index.size
      for ( i <- 0 until len){
        arr += ((index, (values.index(i), values.data(i).toFloat)))
      }
      arr
    }


    val result = thisEmits.join(otherEmits).flatMap{case(index, (valA, valB)) =>
      val arr = new ArrayBuffer[((Long, Long), Float)]()
      for (l <- valA){
        arr += (((l._1, valB._1), l._2 * valB._2))
      }
      arr
    }.reduceByKey( _ + _).map(t=>MatrixEntry(t._1._1,t._1._2,t._2))
    new CoordinateMatrix(result, nRows, other.nCols)
  }

 /** transform the DenseVecMatrix to BlockMatrix
  *
  * @return original matrix in DenseVecMatrix type
  */

  def elementsCount(): Long = {
    rows.count()
  }

}
