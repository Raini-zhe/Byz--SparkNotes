package graph

import org.apache.spark.graphx.{Graph, GraphLoader}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkContext, SparkConf}

import scala.util.Random

/**
 * Created by sara on 10/25/15.
  * * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object BiModularClustering {
  def main(args: Array[String]): Unit ={
    Random.setSeed(System.currentTimeMillis());
    val filename = args(0);
    val conf = new SparkConf().setAppName("modularity");
    val sc = new SparkContext(conf)
    val inputgraph = GraphLoader.edgeListFile(sc,filename)
    val numNode = inputgraph.numVertices.toInt;
    println("Num node=" + numNode);



    val degreeRDD = inputgraph.outDegrees
    val degGraph = Graph(degreeRDD, inputgraph.edges );
    val clusterGraph = inputgraph.mapVertices{ case (id, cluster)=> (0, if (Random.nextDouble() >= 0.5) 1 else -1) }
    val degClusterGraph = clusterGraph.joinVertices(degGraph.vertices){case (id, (_, cluster), degree)=> (degree, cluster)}


    //val clusterGraph = .mapVertices{case (id, (v1,v2)) => if (clusters.predict( Vectors.dense(v2)) >= 1.0) {1} else 2}
    //val group1Ids = group1.vertices.map{case (id,(_,_))=> id }.collect();
    //val group2Ids = group1.vertices.map{case (id,(_,_))=> id }.collect();
    //val f = (x:VertexId) => if (group1Ids.contains(x)) {1} else {2}

    val modularity = new Modularity().evaluate2(degClusterGraph)
    println("modularity = " + modularity)
    for (i<-1 to 20) {
      val cluster_1 = clusterGraph.vertices.filter { case (id, _) => id == i }.collect().apply(0)._2._2;
      val gain = new Modularity().delta(degClusterGraph, i, -1 * cluster_1);
      println("gain " + i + "  = " + gain)
    }
  }



}
