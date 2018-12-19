package graph

import org.apache.spark.graphx.Graph

import scala.util.Random

/**
 * Created by sara on 10/23/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
class RandomClustering {
  def run(inputGraph: Graph[Any, Any], clusterNum: Int): Graph[Int, Any] = {

    val min = 0.0

    val binWidth = 1.0 / clusterNum

    //create a list of lower bounds (e.g for 4 bins it's [0.0, 0.25, 0.5, 0.75])
    val bounds = (0 to clusterNum-1).map { x => min + binWidth * x }.toList

    Random.setSeed(System.currentTimeMillis())
    val randomCluster = inputGraph.mapVertices { case (id, _) =>
        var assignedBin = 0
        val d = Random.nextDouble()
        //find the bin that the random number falls inside
        for (i <-0 to clusterNum-1) {
          if (d >= bounds(i)) {
            assignedBin = i
          }
        }
        assignedBin
    }
    randomCluster
  }
}
