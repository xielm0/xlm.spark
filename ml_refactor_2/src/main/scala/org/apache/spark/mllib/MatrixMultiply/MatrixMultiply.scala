package org.apache.spark.mllib.MatrixMultiply

import com.jd.szad.tools.Writer
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by xieliming on 2017/2/9.
 */

object MatrixMultiply {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("user_label_m")
      .set("spark.akka.timeout", "1000")
      .set("spark.rpc.askTimeout", "500")
     
    val sc = new SparkContext(conf)

    val sqlContext = new org.apache.spark.sql.hive.HiveContext(sc)

    //prepair data
    val sql =
      s"""
         |select uid,concat(type,'_',label) as label, rate
         |from app.app_szad_m_dyrec_userlabel_apply
         |where user_type=1
         |and length(uid)>20
         |and type in ('browse_top20cate')
         |and rn<=20
        """.stripMargin
    val df_user_label = sqlContext.sql(sql)

    val sql3 =
      s"""
         |select concat(type,'_',label) as label,sku,score from app.app_szad_m_dyRec_userlabel_model_2 where type ='browse_top20cate'
        """.stripMargin
    val df_label_sku = sqlContext.sql(sql3)

    val user_label = df_user_label.rdd.map(t => (t.getAs("uid").toString, t.getAs("label").toString, t.getAs("rate").asInstanceOf[Int]))
    val label_sku = df_label_sku.rdd.map(t => (t.getAs("label").toString, t.getAs("sku").toString, t.getAs("score").asInstanceOf[Double])).cache()

    //tranform to int
    val users = user_label.map(_._1).distinct().zipWithIndex() //(user,Long)  val tmp=users.collectAsMap()
    val labels = label_sku.map(_._1).distinct().zipWithIndex().collectAsMap()
    val skus = label_sku.map(_._2).distinct().zipWithIndex().collectAsMap() //drive
    val labels_bc = sc.broadcast(labels)
    val skus_bc = sc.broadcast(skus)

    val n = label_sku.map(_._1).distinct().count()
    println("lables 's count =" + n)

    //tranform to matrix row=uid,col=label
    val A = user_label.map(t => (t._1, (t._2, t._3))).join(users).
      map { t =>
        val user = t._2._2
        if (labels_bc.value.contains(t._2._1._1)) {
          val label = labels_bc.value(t._2._1._1)
          (user, label, t._2._1._2)
        } else {
          (-1L, -1L, 0)
        }
      }.filter(_._1 >= 0)

    val B = label_sku.map { t =>
      val label = labels_bc.value(t._1)
      if (skus_bc.value.contains(t._2)) {
        val sku = skus_bc.value(t._2)
        (label, sku, t._3)
      } else {
        (-1L, -1L, 0D)
      }
    }.filter(_._1 >= 0)

    //params
    val mul_type = args(0)
    if (mul_type == "block") {
      /*
         * 块矩阵相乘
         * shuffle过大，1.5.2的版本容易出现Missing an output location for shuffle
         * */

      //增加一个最大边界
      val A_2 = sc.parallelize(Array(1, n)).map(t => MatrixEntry(1L, t - 1, 0))

      //转换成矩阵
      val mat_A = new CoordinateMatrix(A.map(t => MatrixEntry(t._1, t._2, t._3)) ++ (A_2))
      val mat_B = new CoordinateMatrix(B.map(t => MatrixEntry(t._1, t._2, t._3)))

      val indMat_A = new CoordinateMatrix(A.map(t => MatrixEntry(t._1, t._2, t._3))).toIndexedRowMatrix()
      val indMat_B = new CoordinateMatrix(B.map(t => MatrixEntry(t._1, t._2, t._3))).toIndexedRowMatrix()

      val ind_s = indMat_A.multiply(indMat_B.toBlockMatrix().toLocalMatrix())


      //compute
      // block matrix *  block matrix

      import org.apache.spark.mllib.linalg.distributed2.BlockMatrixEx.BlockMatrixV2
      import org.apache.spark.mllib.linalg.distributed2.CoordinateMatrixEx.CoordinateMatrixV2
      val block_A = mat_A.toBlockMatrixV2(204800, 102400)
      val block_B = mat_B.toBlockMatrixV2(102400, 10240)
      println("A.blocks.partitions.length= " + block_A.blocks.partitions.length + ",A.numRowBlocks=" + block_A.numRowBlocks+ ",A.numColBlocks=" + block_A.numColBlocks)
      println("A.numRows=" + block_A.numRows+ ",A.numCols=" + block_A.numCols)
      println("B.blocks.partitions.length= " + block_B.blocks.partitions.length + ",B.numRowBlocks=" + block_B.numRowBlocks+ ",B.numColBlocks=" + block_B.numColBlocks)
      println("A.numRows=" + block_B.numRows+ ",A.numCols=" + block_B.numCols)

      val S = block_A.multiplyV2(block_B)
      println("S.blocks.partitions.length=" + S.blocks.partitions.length + ",S.numRows=" + S.numRows)

      // top 100
      val top100 = S.toCoordinateMatrix().entries.map(t => (t.i, (t.j, t.value))).groupByKey().flatMap {
        case (a, b) => //b=Interable[(item2,score)]
          val topK = b.toArray.sortWith { (b1, b2) => b1._2 > b2._2 }.take(100)
          //          topk.map{ t => (a,t._1,t._2) }
          topK.zipWithIndex.map { t => (a, (t._1._1, t._1._2, t._2)) } // (user,sku,score,rn)
      }

      //transform to user, sku
      val users_2 = users.map(_.swap) //(long,string)
      val skus_2 = skus.map(_.swap)
      val res = top100.join(users_2).map { case (user_int, ((sku_int, score, rn), user)) => (user, skus_2(sku_int), score, rn) }
        .map(t => t._1 + "\t" + t._2 + "\t" + t._3 + "\t" + t._4)

      //save
      Writer.write_table(res, "app.db/app_szad_m_dyrec_userlabel_predict_res/user_type=1", "lzo")

    }else {
      //(long, Vector)
      val rdd_A= A.map(t=>(t._1,(t._2.toInt,t._3.toDouble))).groupByKey().map{
        case (a,b) => //( )
          val b2=b.toSeq
          val b3 :Vector= Vectors.sparse( n.toInt,b2)
          (a,b3)
      }
      //mat_B要足够小，require(numRows() * numCols() < Int.MaxValue
      val mat_B = new CoordinateMatrix(B.map(t=>MatrixEntry(t._1,t._2,t._3))).toBlockMatrix().toLocalMatrix() // local
      val bc_B = sc.broadcast(mat_B)
      println("size =" +mat_B.numActives)
      val S=rdd_A.flatMap{
        case(uid,v)=>
          val s = bc_B.value.multiply(v).toSparse
          val ind = s.indices
          val r = ind.map(i => (i,s.values(i)))
          //top 100
          val r2 =r.toBuffer
          val topK = r2.sortBy(_._2).reverse
          if (topK.length > 100) topK.remove(100,r.length - 100)
          topK.zipWithIndex.map { t => (uid, (t._1._1, t._1._2, t._2)) }
      }

      //transform to user, sku
      val users_2 = users.map(_.swap) //(long,string)
      val skus_2 = skus.map(_.swap)
      val res = S.join(users_2).map { case (user_int, ((sku_int, score, rn), user)) => (user, skus_2(sku_int), score, rn) }
        .map(t => t._1 + "\t" + t._2 + "\t" + t._3 + "\t" + t._4)

      //save
      Writer.write_table(res, "app.db/app_szad_m_dyrec_userlabel_predict_res/user_type=1", "lzo")

    }

    sc.stop()
  }


}
