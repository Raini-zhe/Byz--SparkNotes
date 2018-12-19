package graph

import org.apache.spark.graphx._
import org.apache.spark.{SparkContext, SparkConf}
import utils.{MetaFile, FileHandler}

/**
 * Created by sara on 12/7/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object LargestCC {
  def main(args: Array[String]) {
    val filename = args(0)
    val objects = FileHandler.loadInput(filename);
    val vertexFile = objects(0).uri
    val edgeFile = objects(1).uri
    val uritype = objects(0).uriType

    val conf = new SparkConf().setAppName("cc");

    val sc = new SparkContext(conf)
    val vertexRDD = sc.objectFile[(VertexId, Int)](vertexFile)
    val edgeRDD = sc.objectFile[org.apache.spark.graphx.Edge[Int]](edgeFile);
    val graph = Graph(vertexRDD, edgeRDD)

    val cc = graph.connectedComponents()
    val sizes =cc.vertices.groupBy(_._2).map((p=>(p._1,p._2.size))).sortBy(x=>x._2, false).collect()

    val largestCCId = sizes.apply(0)._1

    val largestCC = cc.subgraph(vpred = (id, cid) => cid == largestCCId)

    val outVerFile = FileHandler.getFullName(sc, uritype, "vertices")
    val outEdgeFile = FileHandler.getFullName(sc, uritype, "edges")
    //largestCC.vertices
    largestCC.vertices.saveAsObjectFile(outVerFile)
    largestCC.edges.saveAsObjectFile(outEdgeFile)

    val outObjects = new Array[MetaFile](2)
    val verObj = new MetaFile
    verObj.objName = "VertexRDD"
    verObj.uri = outVerFile
    verObj.uriType = uritype

    val edgeObj = new MetaFile
    edgeObj.objName = "EdgeRDD"
    edgeObj.uri = outEdgeFile
    edgeObj.uriType = uritype

    outObjects.update(0, verObj)
    outObjects.update(1, edgeObj)
    FileHandler.dumpOutput(args(1), objects)
  }

}
