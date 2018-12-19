package inout

import org.apache.spark.graphx._
import org.apache.spark.{SparkConf, SparkContext}
import utils._

/**
  * Created by sara on 11/6/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
  */
object Data2Graph {
   def main(args: Array[String]) {

     val filename = args(0)
     val output= args(1)
     val log = args(2)
     val inputType = args(3)
     val uriType = args(4)

     FileLogger.open(log)
     try {

       val start = System.currentTimeMillis();

       val conf = new SparkConf().setAppName("graphLoader")
       val sc = new SparkContext(conf)
       val metaFiles = FileHandler.loadInput(filename)
       val uritype = metaFiles(0).uriType
       val file = metaFiles(0).uri



       if (inputType.equals("edgeview")) {
         val uritype = uriType
         val graph = GraphLoader.edgeListFile(sc, file)
         val outVerFile = FileHandler.getFullName(sc, uritype, "vertices")
         val outEdgeFile = FileHandler.getFullName(sc, uritype, "edges")
         graph.vertices.saveAsObjectFile(outVerFile)
         graph.edges.saveAsObjectFile(outEdgeFile)
         val verObj = new MetaFile("VertexRDD[Any]", uritype, outVerFile)
         val edgeObj = new MetaFile("EdgeRDD[Any]", uritype, outEdgeFile)
         FileHandler.dumpOutput(output, Array(verObj, edgeObj))
         FileLogger.println("Number of verteces: " + graph.vertices.count())
         FileLogger.println("Number of edges: " + graph.edges.count())
       }
       val stop = System.currentTimeMillis()
       FileLogger.println("Data loaded sucessfully in " + ((stop - start) / 1000.0) + " sec")

     } catch {
       case e: Exception => FileLogger.println("ERROR. tool unsuccessful:" + e);
     } finally {
       FileLogger.close()
     }
   }

 }
