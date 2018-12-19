package graph


import org.apache.spark.graphx._
import scala.math._
import utils.FileLogger
/**
 * Created by sara on 10/22/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
class Modularity {

  def delta (g: Graph[(Int,Int),Int], vertexId: VertexId, newCluster: Int): Double = {
    val k_i = g.vertices.filter{case(id, _) => id==vertexId}.collect().apply(0)._2._1;
    val sum_in = g.triplets.map{ t=> if (t.srcAttr._2 == newCluster && t.dstAttr._2 == newCluster) 1.0 else 0.0}.reduce(_+_) / 2;
    val sum_tot = g.triplets.map{ t=> if (t.srcAttr._2 == newCluster || t.dstAttr._2 == newCluster) 1.0 else 0.0}.reduce(_+_) / 2;
    val k_in = g.triplets.map{ t=> if ( (t.srcAttr._2 == newCluster && vertexId == t.dstId) ||  (t.dstAttr._2 == newCluster && vertexId == t.srcId) ) 1.0 else 0.0}.reduce(_+_) / 2;
    val m = g.edges.map({e=>e.attr}).reduce(_+_)
    val diff = (sum_in + k_in) / m - sqrt((sum_tot+k_i) / m) - (sum_in/m) + sqrt (sum_tot/m) + sqrt(k_i/m)
    return diff
  }

  def evaluate(g: Graph[Int, Int]) : Double = {

    //val degreeRDD = g.outDegrees
   // val degGraph = Graph(degreeRDD, g.edges );
    //val clusterGraph = g.mapVertices{ case (id, cluster)=> (0, cluster) } //add place holder as a first attr
    //FileLogger.println(clusterGraph.vertices.collect().mkString(","));
    val start = System.currentTimeMillis();
    g.cache()
    val degClusterVertices = g.aggregateMessages [(Double, Double)](
      sendMsg = t => t.sendToSrc((1.0, t.srcAttr)),
      mergeMsg = (m1, m2) => (m1._1+ m2._1, m1._2)
    )
    val numCluster = g.vertices.reduce{case ((id1,v1), (id2,v2))=> (-1, max(v1, v2))}._2 + 1
    val m = g.numEdges / 2

    val degClusterGraph = Graph(degClusterVertices, g.edges)

    g.unpersist()
    degClusterGraph.cache()
    //val degClusterGraph2 = clusterGraph.joinVertices(degGraph.vertices){
    //          case (id, (_, cluster), degree)=> (degree, cluster)}
    //FileLogger.println(degClusterGraph.vertices.collect().mkString(","));

    //degClusterGraph.vertices.count();

    val stop = System.currentTimeMillis();
    //FileLogger.println("init in " + (stop - start)/ 1000.0 + "sec" )

    //val m = g.edges.map({e=>1.0}).reduce(_+_) / 2
    FileLogger.println("number of edges:"  + m);

    //In degClusterGraph, the attribute of each vertex is
    // a tuple of the vertex degree and its cluster id


    //val product = degClusterGraph.vertices.cartesian(degClusterGraph.vertices)
      //.filter{ case  ((v1, (d1, c1)), (v2,(d2,c2)))=> v1 != v2 && c1 == c2 }
    //val expected = product.map{case ((v1, (d1, c1)), (v2,(d2,c2)))=>
     // d1*d2/(2*m)}.reduce(_+_) / (2*m)




    val existing = degClusterGraph.triplets.map(t=>
      if (t.srcAttr._2 == t.dstAttr._2) {0.5} else {0.0}).reduce(_+_) / (m)
    FileLogger.println("existing " + existing)

    var expected = 0.0
    for (i<-0 to numCluster - 1) {
      val part = degClusterGraph.vertices.map{case (v,(d, c))=>
        if (c == i) {d} else 0.0}.reduce(_+_) / (2*m)
      expected = expected + pow(part,2)
    }
    val modularity = existing - expected

    modularity

  }
  def evaluate2(g: Graph[(Int,Int), Int]): Double = {
    val m = g.edges.map({e=>e.attr}).reduce(_+_)
    val modularity = g.triplets.map({t=>(t.attr - t.srcAttr._1*t.dstAttr._1/m)*( if (t.srcAttr._2 == t.dstAttr._2) {1.0} else {0.0}) }).reduce(_+_) / (m);
    return  modularity
  }
}
