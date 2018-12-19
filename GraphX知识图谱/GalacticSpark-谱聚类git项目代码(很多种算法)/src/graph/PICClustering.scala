package graph

import org.apache.spark.graphx.Graph
import org.apache.spark.mllib.clustering.PowerIterationClustering

/**
  * Created by sara on 1/25/16.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
  */
class PICClustering {
  def run(inputGraph: Graph[Any, Any], clusterNum: Int, iterationNum: Int): Graph[Int, Any] = {


    val similarities = inputGraph.edges.map { e =>
      (e.srcId.toLong, e.dstId.toLong, 1.0)
    }

    val pic = new PowerIterationClustering()
      .setK(clusterNum)
      .setMaxIterations(iterationNum)

    val model = pic.run(similarities)

    val vertexRDD = model.assignments.map{f=>(f.id, f.cluster)}
    val clusterGraph = Graph(vertexRDD, inputGraph.edges)
    clusterGraph
  }
}
