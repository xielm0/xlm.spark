/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.linalg.distributed2

import org.apache.spark.SparkException
import org.apache.spark.mllib.linalg.distributed.BlockMatrix
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrices, Matrix, SparseMatrix}

object BlockMatrixEx extends Serializable
{
  implicit class BlockMatrixV2(bm: BlockMatrix) extends Serializable{
    private type MatrixBlock2 = ((Int, Int), Matrix)

    def multiplyV2(other: BlockMatrix): BlockMatrix = {
      require(bm.numCols() == other.numRows(), "The number of columns of A and the number of rows " +
        s"of B must be equal. A.numCols: ${bm.numCols()}, B.numRows: ${other.numRows()}. If you " +
        "think they should be equal, try setting the dimensions of A and B explicitly while " +
        "initializing them.")
      if (bm.colsPerBlock == other.rowsPerBlock) {
//        val resultPartitioner = GridPartitioner(bm.numRowBlocks , other.numColBlocks,10000)
        val resultPartitioner = new MatrixPartitioner(bm.numRowBlocks, other.numColBlocks,
          bm.rowsPerBlock, bm.colsPerBlock)
        // Each block of A must be multiplied with the corresponding blocks in each column of B.
        // TODO: Optimize to send block to a partition once, similar to ALS
        val flatA = bm.blocks.flatMap { case ((blockRowIndex, blockColIndex), block) =>
          Iterator.tabulate(other.numColBlocks)(j => ((blockRowIndex, j, blockColIndex), block))
        }.partitionBy(resultPartitioner)
        // Each block of B must be multiplied with the corresponding blocks in each row of A.
        val flatB = other.blocks.flatMap { case ((blockRowIndex, blockColIndex), block) =>
          Iterator.tabulate(bm.numRowBlocks)(i => ((i, blockColIndex, blockRowIndex), block))
        }.partitionBy(resultPartitioner)
//        println("test is ok")
        val newBlocks = flatA.cogroup(flatB, resultPartitioner)
          .flatMap { case ((blockRowIndex, blockColIndex, _), (a, b)) =>
            if (a.size > 1 || b.size > 1) {
              throw new SparkException("There are multiple MatrixBlocks with indices: " +
                s"($blockRowIndex, $blockColIndex). Please remove them.")
            }
            if (a.nonEmpty && b.nonEmpty) {
              val C = b.head match {
                case dense: DenseMatrix => a.head.multiply(dense)
                case sparse: SparseMatrix => a.head.multiply(sparse.toDense) // choice it
                case _ => throw new SparkException(s"Unrecognized matrix type ${b.head.getClass}.")
              }
              Iterator(((blockRowIndex, blockColIndex), C.toBreeze))
            } else {
              Iterator()
            }
          }.reduceByKey(resultPartitioner, (a, b) => a + b)
          .mapValues(Matrices.fromBreeze)
        // TODO: Try to use aggregateByKey instead of reduceByKey to get rid of intermediate matrices
        new BlockMatrix(newBlocks, bm.rowsPerBlock, other.colsPerBlock, bm.numRows(), other.numCols())
      } else {
        throw new SparkException("colsPerBlock of A doesn't match rowsPerBlock of B. " +
          s"A.colsPerBlock: $bm.colsPerBlock, B.rowsPerBlock: ${other.rowsPerBlock}")
      }
    }

    private type BlockDestinations = Map[(Int, Int), Set[Int]]
    private lazy val blockInfo = bm.blocks.mapValues(block => (block.numRows, block.numCols)).cache()

    // 左矩阵中列号会和右矩阵行号相同的块相乘，得到左矩阵中列索引 与 右矩阵中行索引 相同的set。
    // 由于有这个判断，右矩阵中没有值的快左矩阵就不会重复复制了，避免了零值计算。
    private def simulateMultiply(
                                   other: BlockMatrix,
                                   partitioner: MatrixPartitioner): (BlockDestinations, BlockDestinations) = {
      val leftMatrix = blockInfo.keys.collect() //
      val rightMatrix = other.blocks.keys.collect()

      val rightCounterpartsHelper = rightMatrix.groupBy(_._1).mapValues(_.map(_._2))
      val leftDestinations = leftMatrix.map { case (rowIndex, colIndex) =>
        val rightCounterparts = rightCounterpartsHelper.getOrElse(colIndex, Array())
        val partitions = rightCounterparts.map(b => partitioner.getPartition((rowIndex, b)))
        ((rowIndex, colIndex), partitions.toSet)
      }.toMap

      val leftCounterpartsHelper = leftMatrix.groupBy(_._2).mapValues(_.map(_._1))
      val rightDestinations = rightMatrix.map { case (rowIndex, colIndex) =>
        val leftCounterparts = leftCounterpartsHelper.getOrElse(rowIndex, Array())
        val partitions = leftCounterparts.map(b => partitioner.getPartition((b, colIndex)))
        ((rowIndex, colIndex), partitions.toSet)
      }.toMap

      (leftDestinations, rightDestinations)
    }

    private def simulateMultiply2(
                                     other: BlockMatrix,
                                     partitioner: MatrixPartitioner): (BlockDestinations,BlockDestinations) = {
      val leftMatrix = blockInfo.keys.collect() // blockInfo should already be cached
      val rightMatrix = other.blocks.keys.collect()
      //以下这段代码这样理解，假设A*B=C,因为A11在计算C11到C1n的时候会用到，所以A11在计算C11到C1n的机器都会存放一份。
      val leftDestinations = leftMatrix.map {case(rowIndex,colIndex) =>
        //左矩阵中列号会和右矩阵行号相同的块相乘，得到所有右矩阵中行索引和左矩阵中列索引相同的矩阵的位置。
        // 由于有这个判断，右矩阵中没有值的快左矩阵就不会重复复制了，避免了零值计算。
        val rightCounterparts = rightMatrix.filter(_._1 == colIndex)
        // 因为矩阵乘完之后还有相加的操作(reduceByKey),相加的操作如果在同一部机器上可以用combineBy进行优化，
        // 这里直接得到每一个分块在进行完乘法之后会在哪些partition中用到。
        val partitions = rightCounterparts.map(b => partitioner.getPartition((rowIndex,b._2)))
        ((rowIndex, colIndex),partitions.toSet)
      }.toMap
      val rightDestinations = rightMatrix.map {case(rowIndex,colIndex) =>
        val leftCounterparts = leftMatrix.filter(_._2 == rowIndex)
        val partitions = leftCounterparts.map(b => partitioner.getPartition((b._1,colIndex)))
        ((rowIndex, colIndex),partitions.toSet)
      }.toMap
      (leftDestinations, rightDestinations)
    }

    def multiplyV3(other: BlockMatrix): BlockMatrix = {
      require(bm.numCols() == other.numRows(), "The number of columns of A and the number of rows " +
        s"of B must be equal. A.numCols: ${bm.numCols()}, B.numRows: ${other.numRows()}. If you " +
        "think they should be equal, try setting the dimensions of A and B explicitly while " +
        "initializing them.")
      if (bm.colsPerBlock == other.rowsPerBlock) {
//        val resultPartitioner =GridPartitioner(bm.numRowBlocks, other.numColBlocks,
//          math.max(bm.blocks.partitions.length, other.blocks.partitions.length))
        val resultPartitioner = new MatrixPartitioner(bm.numRowBlocks, other.numColBlocks,
          bm.rowsPerBlock, bm.colsPerBlock)
        val (leftDestinations, rightDestinations) = simulateMultiply(other, resultPartitioner)
        // Each block of A must be multiplied with the corresponding blocks in the columns of B.
        val flatA = bm.blocks.flatMap { case ((blockRowIndex, blockColIndex), block) =>
          val destinations = leftDestinations.getOrElse((blockRowIndex, blockColIndex), Set.empty)
          destinations.map(j => (j, (blockRowIndex, blockColIndex, block)))
        }
        // Each block of B must be multiplied with the corresponding blocks in each row of A.
        val flatB = other.blocks.flatMap { case ((blockRowIndex, blockColIndex), block) =>
          val destinations = rightDestinations.getOrElse((blockRowIndex, blockColIndex), Set.empty)
          destinations.map(j => (j, (blockRowIndex, blockColIndex, block)))
        }
        val newBlocks = flatA.cogroup(flatB, resultPartitioner).flatMap { case (pId, (a, b)) =>
          a.flatMap { case (leftRowIndex, leftColIndex, leftBlock) =>
            b.filter(_._1 == leftColIndex).map { case (rightRowIndex, rightColIndex, rightBlock) =>
              val C = rightBlock match {
                case dense: DenseMatrix => leftBlock.multiply(dense)
                case sparse: SparseMatrix => leftBlock.multiply(sparse.toDense)
                case _ =>
                  throw new SparkException(s"Unrecognized matrix type ${rightBlock.getClass}.")
              }
              ((leftRowIndex, rightColIndex), C.toBreeze)
            }
          }
        }.reduceByKey(resultPartitioner, (a, b) => a + b).mapValues(Matrices.fromBreeze)
        // TODO: Try to use aggregateByKey instead of reduceByKey to get rid of intermediate matrices
        new BlockMatrix(newBlocks, bm.rowsPerBlock, other.colsPerBlock, bm.numRows(), other.numCols())
      } else {
        throw new SparkException("colsPerBlock of A doesn't match rowsPerBlock of B. " +
          s"A.colsPerBlock: $bm.colsPerBlock, B.rowsPerBlock: ${other.rowsPerBlock}")
      }
    }


  }

}



