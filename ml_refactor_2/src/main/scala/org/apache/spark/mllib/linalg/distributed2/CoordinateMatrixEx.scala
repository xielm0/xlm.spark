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

import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix}
import org.apache.spark.mllib.linalg.{Matrix, SparseMatrix}
import org.apache.spark.rdd.RDD

object CoordinateMatrixEx extends Serializable
{
  implicit class CoordinateMatrixV2(coo: CoordinateMatrix) {
    def toBlockMatrixV2(rowsPerBlock: Int, colsPerBlock: Int): BlockMatrix = {
      require(rowsPerBlock > 0,
        s"rowsPerBlock needs to be greater than 0. rowsPerBlock: $rowsPerBlock")
      require(colsPerBlock > 0,
        s"colsPerBlock needs to be greater than 0. colsPerBlock: $colsPerBlock")
      val m = coo.numRows()
      val n = coo.numCols()
      val numRowBlocks = math.ceil(m.toDouble / rowsPerBlock).toInt  //分成多少RowBlocks
      val numColBlocks = math.ceil(n.toDouble / colsPerBlock).toInt
//      val partitioner = GridPartitioner(numRowBlocks, numColBlocks, coo.entries.partitions.length)
      val partitioner = new MatrixPartitioner(numRowBlocks,numColBlocks, rowsPerBlock,colsPerBlock)

      //coo to block
      val blocks: RDD[((Int, Int), Matrix)] = coo.entries.map { entry =>
        val blockRowIndex = (entry.i / rowsPerBlock).toInt
        val blockColIndex = (entry.j / colsPerBlock).toInt
        val rowId = entry.i % rowsPerBlock  //余数，在submatrix中的位置。
        val colId = entry.j % colsPerBlock

        ((blockRowIndex, blockColIndex), (rowId.toInt, colId.toInt, entry.value))

      }.groupByKey(partitioner).map { case ((blockRowIndex, blockColIndex), entry) =>
        // 当不是行或列最后一块，effRows=rowsPerBlock
        val effRows = math.min(m - blockRowIndex.toLong * rowsPerBlock, rowsPerBlock).toInt
        val effCols = math.min(n - blockColIndex.toLong * colsPerBlock, colsPerBlock).toInt
        ((blockRowIndex, blockColIndex), SparseMatrix.fromCOO(effRows, effCols, entry))
      }
      new BlockMatrix(blocks, rowsPerBlock, colsPerBlock, m, n)
      //blocks 的格式是The RDD of sub-matrix blocks ((blockRowIndex, blockColIndex), sub-matrix)
    }
  }
}



