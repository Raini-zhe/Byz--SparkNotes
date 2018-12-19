package dataset

import org.apache.spark.graphx.{Graph, Edge}
import org.apache.spark.{SparkContext, SparkConf}
import utils.{MetaFile, FileHandler, FileLogger}

import scala.collection.mutable


/**
 * Created by sara on 1/23/16.
  * * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object WikiPageLink {

  def main (args: Array[String]): Unit = {
    val filepath = args(0)
    val output = args(1)
    val log = args(2);
    val uriType = args(3)

    FileLogger.open(log)
    try {
      val conf = new SparkConf().setAppName("lr")
      val sc = new SparkContext(conf)

      val pageLinks = sc.textFile(filepath)
      val uriEdge = pageLinks.filter(l=> (!l.startsWith("#"))).map{l=>val uris = l.split(" "); (uris.apply(0), uris.apply(1))}
      val uriHead = uriEdge.map(e=>e._1)
      val uriTail = uriEdge.map(e=>e._2)
      val allURIs = uriHead.union(uriTail).distinct().zipWithUniqueId()
      val uriMapping = mutable.HashMap.empty[String,Long]
      allURIs.collect().foreach{case (uri, id)=> uriMapping += (uri->id)}
      val edgeRDD = uriEdge.map{case (h,t)=> Edge(uriMapping.get(h).get, uriMapping.get(h).get, 1 ) }
      val vertexRDD = allURIs.map{_.swap}
      val g = Graph(vertexRDD, edgeRDD)
      FileLogger.println("Number of verteces: " + vertexRDD.count())
      FileLogger.println("Number of edges: " + edgeRDD.count() )

      val outVerFile = FileHandler.getFullName(sc, uriType, "vertices")
      val outEdgeFile = FileHandler.getFullName(sc, uriType, "edges")
      //largestCC.vertices
      vertexRDD.saveAsObjectFile(outVerFile)
      edgeRDD.saveAsObjectFile(outEdgeFile)

      val outObjects = new Array[MetaFile](2)
      val verObj = new MetaFile("VertexRDD[(Long,String)]", uriType, outVerFile)
      val edgeObj = new MetaFile("EdgeRDD[Int]", uriType, outEdgeFile)

      FileHandler.dumpOutput(args(1), Array(verObj, edgeObj))

    } catch {
      case e: Exception => FileLogger.println("ERROR. tool unsuccessful:" + e);
    } finally {
      FileLogger.close();
    }
  }
}
