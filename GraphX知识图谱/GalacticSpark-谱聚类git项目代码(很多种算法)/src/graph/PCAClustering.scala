package graph

import org.apache.spark.SparkContext
import org.apache.spark.graphx.{EdgeDirection, Edge, Graph}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Matrix, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD
import scala.collection.mutable

/**
  * Created by sara on 1/25/16.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
  */
class PCAClustering {
  def matrixToRDD(sc:SparkContext, m: Matrix): RDD[Vector] = {
    val columns = m.toArray.grouped(m.numRows)
    val rows = columns.toSeq.transpose // Skip this if you want a column-major RDD.
    val vectors = rows.map(row => new DenseVector(row.toArray))
    sc.parallelize(vectors)
  }

  def run(inputGraph: Graph[Any, Any], clusterNum: Int, eigsNum: Int,sc:SparkContext ): Graph[Int, Any] = {
    val numNode = inputGraph.numVertices.toInt
    val mapping = new mutable.HashMap[Long,Int]()
    val revMapping = new mutable.HashMap[Int, Long]()

    val verticeIds = inputGraph.vertices.map( u => u._1 ).collect()
    for(i<-0 to numNode - 1) {
      mapping.put(verticeIds.apply(i), i)
      revMapping.put(i, verticeIds.apply(i))
    }

    //reindex the verteces from 0 to the num of nodes
    val nVertices = inputGraph.vertices.map( u=> (mapping.apply(u._1).toLong, u._2))
    val nEdges = inputGraph.edges.map(e=> Edge(mapping.apply(e.srcId).toLong, mapping.apply(e.dstId).toLong, e.attr))
    val ngraph = Graph(nVertices, nEdges)

    val output = ngraph.collectNeighborIds(EdgeDirection.Out)
    val spvec = output.mapValues(r => Vectors.sparse( numNode,  r.map(e=>e.toInt) , r.map(e=> 1.0/r.length )))
    val rows = spvec.map(v=>v._2)
    val order = spvec.map(v=>v._1)
    val mat = new RowMatrix(rows)

    val pc = mat.computePrincipalComponents(eigsNum)


    val pcRDD = matrixToRDD(sc, pc)
    val clusters = KMeans.train(pcRDD, clusterNum, 100)

    val clusterArray = pcRDD.map(p=> clusters.predict(p) ).collect()
    val assignedClusters = order.map( o => (o, clusterArray.apply(o.toInt)))
    val origVerextRDD = assignedClusters.map{case (vid, value)=> (revMapping.apply(vid.toInt), value)}
    Graph(origVerextRDD, inputGraph.edges)

  }

}
