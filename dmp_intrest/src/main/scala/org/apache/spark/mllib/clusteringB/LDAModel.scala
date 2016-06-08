package org.apache.spark.mllib.clusteringB

/**
 * Created by xieliming on 2015/11/9.
 */
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, argtopk, normalize, sum => brzSum}
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.annotation.Experimental
import org.apache.spark.api.java.JavaPairRDD
import org.apache.spark.graphx.{Edge, EdgeContext, Graph, VertexId}
import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vector, Vectors}
import org.apache.spark.mllib.util.Loader
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.util.BoundedPriorityQueue
import org.json4s.DefaultFormats
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

/**
 * :: Experimental ::
 *
 * Latent Dirichlet Allocation (LDA) model.
 *
 * This abstraction permits for different underlying representations,
 * including local and distributed data structures.
 */
@Experimental
abstract class LDAModel private[clusteringB]{

  /** Number of topics */
  def k: Int

  /** Vocabulary size (number of terms or terms in the vocabulary) */
  def vocabSize: Int

  /**
   * Inferred topics, where each topic is represented by a distribution over terms.
   * This is a matrix of size vocabSize x k, where each column is a topic.
   * No guarantees are given about the ordering of the topics.
   */
  def topicsMatrix: Matrix

  /**
   * Return the topics described by weighted terms.
   *
   * This limits the number of terms per topic.
   * This is approximate; it may not return exactly the top-weighted terms for each topic.
   * To get a more precise set of top terms, increase maxTermsPerTopic.
   *
   * @param maxTermsPerTopic  Maximum number of terms to collect for each topic.
   * @return  Array over topics.  Each topic is represented as a pair of matching arrays:
   *          (term indices, term weights in topic).
   *          Each topic's terms are sorted in order of decreasing weight.
   */
  def describeTopics(maxTermsPerTopic: Int): Array[(Array[Int], Array[Double])]

  /**
   * Return the topics described by weighted terms.
   *
   * WARNING: If vocabSize and k are large, this can return a large object!
   *
   * @return  Array over topics.  Each topic is represented as a pair of matching arrays:
   *          (term indices, term weights in topic).
   *          Each topic's terms are sorted in order of decreasing weight.
   */
  def describeTopics(): Array[(Array[Int], Array[Double])] = describeTopics(vocabSize)

  /* TODO (once LDA can be trained with Strings or given a dictionary)
   * Return the topics described by weighted terms.
   *
   * This is similar to [[describeTopics()]] but returns String values for terms.
   * If this model was trained using Strings or was given a dictionary, then this method returns
   * terms as text.  Otherwise, this method returns terms as term indices.
   *
   * This limits the number of terms per topic.
   * This is approximate; it may not return exactly the top-weighted terms for each topic.
   * To get a more precise set of top terms, increase maxTermsPerTopic.
   *
   * @param maxTermsPerTopic  Maximum number of terms to collect for each topic.
   * @return  Array over topics.  Each topic is represented as a pair of matching arrays:
   *          (terms, term weights in topic) where terms are either the actual term text
   *          (if available) or the term indices.
   *          Each topic's terms are sorted in order of decreasing weight.
   */
  // def describeTopicsAsStrings(maxTermsPerTopic: Int): Array[(Array[Double], Array[String])]

  /* TODO (once LDA can be trained with Strings or given a dictionary)
   * Return the topics described by weighted terms.
   *
   * This is similar to [[describeTopics()]] but returns String values for terms.
   * If this model was trained using Strings or was given a dictionary, then this method returns
   * terms as text.  Otherwise, this method returns terms as term indices.
   *
   * WARNING: If vocabSize and k are large, this can return a large object!
   *
   * @return  Array over topics.  Each topic is represented as a pair of matching arrays:
   *          (terms, term weights in topic) where terms are either the actual term text
   *          (if available) or the term indices.
   *          Each topic's terms are sorted in order of decreasing weight.
   */
  // def describeTopicsAsStrings(): Array[(Array[Double], Array[String])] =
  //  describeTopicsAsStrings(vocabSize)

  /* TODO
   * Compute the log likelihood of the observed tokens, given the current parameter estimates:
   *  log P(docs | topics, topic distributions for docs, alpha, eta)
   *
   * Note:
   *  - This excludes the prior.
   *  - Even with the prior, this is NOT the same as the data log likelihood given the
   *    hyperparameters.
   *
   * @param documents  RDD of documents, which are term (word) count vectors paired with IDs.
   *                   The term count vectors are "bags of words" with a fixed-size vocabulary
   *                   (where the vocabulary size is the length of the vector).
   *                   This must use the same vocabulary (ordering of term counts) as in training.
   *                   Document IDs must be unique and >= 0.
   * @return  Estimated log likelihood of the data under this model
   */
  // def logLikelihood(documents: RDD[(Long, Vector)]): Double

  /* TODO
   * Compute the estimated topic distribution for each document.
   * This is often called 'theta' in the literature.
   *
   * @param documents  RDD of documents, which are term (word) count vectors paired with IDs.
   *                   The term count vectors are "bags of words" with a fixed-size vocabulary
   *                   (where the vocabulary size is the length of the vector).
   *                   This must use the same vocabulary (ordering of term counts) as in training.
   *                   Document IDs must be unique and >= 0.
   * @return  Estimated topic distribution for each document.
   *          The returned RDD may be zipped with the given RDD, where each returned vector
   *          is a multinomial distribution over topics.
   */
  // def topicDistributions(documents: RDD[(Long, Vector)]): RDD[(Long, Vector)]


}

/**
 * :: Experimental ::
 *
 * Local LDA model.
 * This model stores only the inferred topics.
 * It may be used for computing topics for new documents, but it may give less accurate answers
 * than the [[DistributedLDAModel]].
 *
 * @param topics Inferred topics (vocabSize x k matrix).
 */
@Experimental
class LocalLDAModel private[clusteringB] (
                                          private val topics: Matrix ) extends LDAModel with Serializable {

  override def k: Int = topics.numCols

  override def vocabSize: Int = topics.numRows

  override def topicsMatrix: Matrix = topics

  override def describeTopics(maxTermsPerTopic: Int): Array[(Array[Int], Array[Double])] = {
    val brzTopics = topics.toBreeze.toDenseMatrix
    Range(0, k).map { topicIndex =>
      val topic = normalize(brzTopics(::, topicIndex), 1.0)
      val (termWeights, terms) =
        topic.toArray.zipWithIndex.sortBy(-_._1).take(maxTermsPerTopic).unzip
      (terms.toArray, termWeights.toArray)
    }.toArray
  }

  // TODO
  // override def logLikelihood(documents: RDD[(Long, Vector)]): Double = ???

  // TODO:
  // override def topicDistributions(documents: RDD[(Long, Vector)]): RDD[(Long, Vector)] = ???

}

/**
 * :: Experimental ::
 *
 * Distributed LDA model.
 * This model stores the inferred topics, the full training dataset, and the topic distributions.
 * When computing topics for new documents, it may give more accurate answers
 * than the [[LocalLDAModel]].
 */

@Experimental
class DistributedLDAModel private (
                                    private val graph: Graph[LDA.TopicCounts, LDA.TokenCount],
                                    private val globalTopicTotals: LDA.TopicCounts,
                                    val k: Int,
                                    val vocabSize: Int,
                                     val docConcentration: Double,
                                     val topicConcentration: Double,
                                    private[spark] val iterationTimes: Array[Double]) extends LDAModel {

  import LDA._

  private[clusteringB] def this(state: EMLDAOptimizer, iterationTimes: Array[Double]) = {
    this(state.graph, state.globalTopicTotals, state.k, state.vocabSize, state.docConcentration,
      state.topicConcentration, iterationTimes)
  }

  /**
   * Convert model to a local model.
   * The local model stores the inferred topics but not the topic distributions for training
   * documents.
   */
  def toLocal: LocalLDAModel = new LocalLDAModel(topicsMatrix )

  /**
   * Inferred topics, where each topic is represented by a distribution over terms.
   * This is a matrix of size vocabSize x k, where each column is a topic.
   * No guarantees are given about the ordering of the topics.
   *
   * WARNING: This matrix is collected from an RDD. Beware memory usage when vocabSize, k are large.
   */
  override lazy val topicsMatrix: Matrix = {
    // Collect row-major topics
    val termTopicCounts: Array[(Int, TopicCounts)] =
      graph.vertices.filter(_._1 < 0).map { case (termIndex, cnts) =>
        (index2term(termIndex), cnts)
      }.collect()
    // Convert to Matrix
    val brzTopics = BDM.zeros[Double](vocabSize, k)
    termTopicCounts.foreach { case (term, cnts) =>
      var j = 0
      while (j < k) {
        brzTopics(term, j) = cnts(j)
        j += 1
      }
    }
    Matrices.fromBreeze(brzTopics)
  }

  override def describeTopics(maxTermsPerTopic: Int): Array[(Array[Int], Array[Double])] = {
    val numTopics = k
    // Note: N_k is not needed to find the top terms, but it is needed to normalize weights
    //       to a distribution over terms.
    val N_k: TopicCounts = globalTopicTotals
    val topicsInQueues: Array[BoundedPriorityQueue[(Double, Int)]] =
      graph.vertices.filter(isTermVertex)
        .mapPartitions { termVertices =>
          // For this partition, collect the most common terms for each topic in queues:
          //  queues(topic) = queue of (term weight, term index).
          // Term weights are N_{wk} / N_k.
          val queues =
            Array.fill(numTopics)(new BoundedPriorityQueue[(Double, Int)](maxTermsPerTopic))
          for ((termId, n_wk) <- termVertices) {
            var topic = 0
            while (topic < numTopics) {
              queues(topic) += (n_wk(topic) / N_k(topic) -> index2term(termId.toInt))
              topic += 1
            }
          }
          Iterator(queues)
        }.reduce { (q1, q2) =>
        q1.zip(q2).foreach { case (a, b) => a ++= b}
        q1
      }
    topicsInQueues.map { q =>
      val (termWeights, terms) = q.toArray.sortBy(-_._1).unzip
      (terms.toArray, termWeights.toArray)
    }
  }

  // TODO
  // override def logLikelihood(documents: RDD[(Long, Vector)]): Double = ???

  /**
   * Log likelihood of the observed tokens in the training set,
   * given the current parameter estimates:
   *  log P(docs | topics, topic distributions for docs, alpha, eta)
   *
   * Note:
   *  - This excludes the prior; for that, use [[logPrior]].
   *  - Even with [[logPrior]], this is NOT the same as the data log likelihood given the
   *    hyperparameters.
   */
  lazy val logLikelihood: Double = {
    val eta = topicConcentration
    val alpha = docConcentration
    assert(eta > 1.0)
    assert(alpha > 1.0)
    val N_k = globalTopicTotals
    val smoothed_N_k: TopicCounts = N_k + (vocabSize * (eta - 1.0))
    // Edges: Compute token log probability from phi_{wk}, theta_{kj}.
    val sendMsg: EdgeContext[TopicCounts, TokenCount, Double] => Unit = (edgeContext) => {
      val N_wj = edgeContext.attr
      val smoothed_N_wk: TopicCounts = edgeContext.dstAttr + (eta - 1.0)
      val smoothed_N_kj: TopicCounts = edgeContext.srcAttr + (alpha - 1.0)
      val phi_wk: TopicCounts = smoothed_N_wk :/ smoothed_N_k
      val theta_kj: TopicCounts = normalize(smoothed_N_kj, 1.0)
      val tokenLogLikelihood = N_wj * math.log(phi_wk.dot(theta_kj))
      edgeContext.sendToDst(tokenLogLikelihood)
    }
    graph.aggregateMessages[Double](sendMsg, _ + _)
      .map(_._2).fold(0.0)(_ + _)
  }

  /**
   * Log probability of the current parameter estimate:
   *  log P(topics, topic distributions for docs | alpha, eta)
   */
  lazy val logPrior: Double = {
    val eta = topicConcentration
    val alpha = docConcentration
    // Term vertices: Compute phi_{wk}.  Use to compute prior log probability.
    // Doc vertex: Compute theta_{kj}.  Use to compute prior log probability.
    val N_k = globalTopicTotals
    val smoothed_N_k: TopicCounts = N_k + (vocabSize * (eta - 1.0))
    val seqOp: (Double, (VertexId, TopicCounts)) => Double = {
      case (sumPrior: Double, vertex: (VertexId, TopicCounts)) =>
        if (isTermVertex(vertex)) {
          val N_wk = vertex._2
          val smoothed_N_wk: TopicCounts = N_wk + (eta - 1.0)
          val phi_wk: TopicCounts = smoothed_N_wk :/ smoothed_N_k
          (eta - 1.0) * brzSum(phi_wk.map(math.log))
        } else {
          val N_kj = vertex._2
          val smoothed_N_kj: TopicCounts = N_kj + (alpha - 1.0)
          val theta_kj: TopicCounts = normalize(smoothed_N_kj, 1.0)
          (alpha - 1.0) * brzSum(theta_kj.map(math.log))
        }
    }
    graph.vertices.aggregate(0.0)(seqOp, _ + _)
  }

  /**
   * For each document in the training set, return the distribution over topics for that document
   * ("theta_doc").
   *
   * @return  RDD of (document ID, topic distribution) pairs
   */
  def topicDistributions: RDD[(Long, Vector)] = {
    graph.vertices.filter(LDA.isDocumentVertex).map { case (docID, topicCounts) =>
      (docID.toLong, Vectors.fromBreeze(normalize(topicCounts, 1.0)))
    }
  }

  /** Java-friendly version of [[topicDistributions]] */
  def javaTopicDistributions: JavaPairRDD[java.lang.Long, Vector] = {
    JavaPairRDD.fromRDD(topicDistributions.asInstanceOf[RDD[(java.lang.Long, Vector)]])
  }

  // TODO:
  // override def topicDistributions(documents: RDD[(Long, Vector)]): RDD[(Long, Vector)] = ???

  def topDocumentsPerTopic(maxDocumentsPerTopic: Int): Array[(Array[Long], Array[Double])] = {
    val numTopics = k
    val topicsInQueues: Array[BoundedPriorityQueue[(Double, Long)]] =
      topicDistributions.mapPartitions { docVertices =>
        // For this partition, collect the most common docs for each topic in queues:
        //  queues(topic) = queue of (doc topic, doc ID).
        val queues =
          Array.fill(numTopics)(new BoundedPriorityQueue[(Double, Long)](maxDocumentsPerTopic))
        for ((docId, docTopics) <- docVertices) {
          var topic = 0
          while (topic < numTopics) {
            queues(topic) += (docTopics(topic) -> docId)
            topic += 1
          }
        }
        Iterator(queues)
      }.treeReduce { (q1, q2) =>
        q1.zip(q2).foreach { case (a, b) => a ++= b }
        q1
      }
    topicsInQueues.map { q =>
      val (docTopics, docs) = q.toArray.sortBy(-_._1).unzip
      (docs.toArray, docTopics.toArray)
    }
  }
  /**
   * For each document, return the top k weighted topics for that document and their weights.
   * @return RDD of (doc ID, topic indices, topic weights)
   */
  def topTopicsPerDocument(k: Int): RDD[(Long, Array[Int], Array[Double])] = {
    graph.vertices.filter(LDA.isDocumentVertex).map { case (docID, topicCounts) =>
      val topIndices = argtopk(topicCounts, k)
      val sumCounts = brzSum(topicCounts)
      val weights = if (sumCounts != 0) {
        topicCounts(topIndices) / sumCounts
      } else {
        topicCounts(topIndices)
      }
      (docID.toLong, topIndices.toArray, weights.toArray)
    }
  }

   def save(sc: SparkContext, path: String): Unit = {
    DistributedLDAModel.SaveLoadV1_0.save(
      sc, path, graph, globalTopicTotals, k, vocabSize, docConcentration, topicConcentration,
      iterationTimes /*, gammaShape*/)
  }


}

object DistributedLDAModel  {

  private[clusteringB] object SaveLoadV1_0 {

    val thisFormatVersion = "1.0"

    val thisClassName = "org.apache.spark.mllib.clusteringB.DistributedLDAModel"

    // Store globalTopicTotals as a Vector.
    case class Data(globalTopicTotals: Vector)

    // Store each term and document vertex with an id and the topicWeights.
    case class VertexData(id: Long, topicWeights: Vector)

    // Store each edge with the source id, destination id and tokenCounts.
    case class EdgeData(srcId: Long, dstId: Long, tokenCounts: Double)

    def save(
              sc: SparkContext,
              path: String,
              graph: Graph[LDA.TopicCounts, LDA.TokenCount],
              globalTopicTotals: LDA.TopicCounts,
              k: Int,
              vocabSize: Int,
              docConcentration: Double,
              topicConcentration: Double,
              iterationTimes: Array[Double] ): Unit = {
      val sqlContext = new org.apache.spark.sql.SQLContext(sc) //SQLContext.getOrCreate(sc)
      import sqlContext.implicits._

      val metadata = compact(render
      (("class" -> thisClassName) ~ ("version" -> thisFormatVersion) ~
        ("k" -> k) ~ ("vocabSize" -> vocabSize) ~
        //("docConcentration" -> docConcentration.toArray.toSeq) ~
        ("docConcentration" -> docConcentration) ~
        ("topicConcentration" -> topicConcentration) ~
        ("iterationTimes" -> iterationTimes.toSeq) ))
      sc.parallelize(Seq(metadata), 1).saveAsTextFile(Loader.metadataPath(path))

      val newPath = new Path(Loader.dataPath(path), "globalTopicTotals").toUri.toString
      sc.parallelize(Seq(Data(Vectors.fromBreeze(globalTopicTotals)))).toDF()
        .save(newPath,"parquet")

      val verticesPath = new Path(Loader.dataPath(path), "topicCounts").toUri.toString
      graph.vertices.map { case (ind, vertex) =>
        VertexData(ind, Vectors.fromBreeze(vertex))
      }.toDF().save(verticesPath,"parquet")

      val edgesPath = new Path(Loader.dataPath(path), "tokenCounts").toUri.toString
      graph.edges.map { case Edge(srcId, dstId, prop) =>
        EdgeData(srcId, dstId, prop)
      }.toDF().save(edgesPath,"parquet")
    }

    def load(
              sc: SparkContext,
              path: String,
              vocabSize: Int,
              docConcentration: Double,
              topicConcentration: Double,
              iterationTimes: Array[Double] ): DistributedLDAModel = {
      val dataPath = new Path(Loader.dataPath(path), "globalTopicTotals").toUri.toString
      val vertexDataPath = new Path(Loader.dataPath(path), "topicCounts").toUri.toString
      val edgeDataPath = new Path(Loader.dataPath(path), "tokenCounts").toUri.toString
      val sqlContext = new org.apache.spark.sql.SQLContext(sc)
      val dataFrame =  sqlContext.load(dataPath,"parquet") //sqlContext.read.parquet(dataPath)
      val vertexDataFrame = sqlContext.load(vertexDataPath,"parquet")
      val edgeDataFrame =  sqlContext.load(edgeDataPath,"parquet")

      Loader.checkSchema[Data](dataFrame.schema)
      Loader.checkSchema[VertexData](vertexDataFrame.schema)
      Loader.checkSchema[EdgeData](edgeDataFrame.schema)
      val globalTopicTotals: LDA.TopicCounts =
        dataFrame.first().getAs[Vector](0).toBreeze.toDenseVector
      val vertices: RDD[(VertexId, LDA.TopicCounts)] = vertexDataFrame.map {
        case Row(ind: Long, vec: Vector) => (ind, vec.toBreeze.toDenseVector)
      }

      val edges: RDD[Edge[LDA.TokenCount]] = edgeDataFrame.map {
        case Row(srcId: Long, dstId: Long, prop: Double) => Edge(srcId, dstId, prop)
      }
      val graph: Graph[LDA.TopicCounts, LDA.TokenCount] = Graph(vertices, edges)

      new DistributedLDAModel(graph, globalTopicTotals, globalTopicTotals.length, vocabSize,
        docConcentration, topicConcentration, iterationTimes )
    }

    def load2(sc: SparkContext, path: String): DistributedLDAModel = {
      val (loadedClassName, loadedVersion, metadata) = Loader.loadMetadata(sc, path)
      implicit val formats = DefaultFormats
      val expectedK = (metadata \ "k").extract[Int]
      val vocabSize = (metadata \ "vocabSize").extract[Int]
      val docConcentration = (metadata \ "docConcentration").extract[Double]
      val topicConcentration = (metadata \ "topicConcentration").extract[Double]
      val iterationTimes = (metadata \ "iterationTimes").extract[Seq[Double]]
      //val gammaShape = (metadata \ "gammaShape").extract[Double]
      val classNameV1_0 = SaveLoadV1_0.thisClassName

      val model = (loadedClassName, loadedVersion) match {
        case (className, "1.0") if className == classNameV1_0 =>
          DistributedLDAModel.SaveLoadV1_0.load(sc, path, vocabSize, docConcentration,
            topicConcentration, iterationTimes.toArray)
        case _ => throw new Exception(
          s"DistributedLDAModel.load did not recognize model with (className, format version):" +
            s"($loadedClassName, $loadedVersion).  Supported: ($classNameV1_0, 1.0)")
      }

      require(model.vocabSize == vocabSize,
        s"DistributedLDAModel requires $vocabSize vocabSize, got ${model.vocabSize} vocabSize")
      require(model.docConcentration == docConcentration,
        s"DistributedLDAModel requires $docConcentration docConcentration, " +
          s"got ${model.docConcentration} docConcentration")
      require(model.topicConcentration == topicConcentration,
        s"DistributedLDAModel requires $topicConcentration docConcentration, " +
          s"got ${model.topicConcentration} docConcentration")
      require(expectedK == model.k,
        s"DistributedLDAModel requires $expectedK topics, got ${model.k} topics")
      model
    }

  }



}
