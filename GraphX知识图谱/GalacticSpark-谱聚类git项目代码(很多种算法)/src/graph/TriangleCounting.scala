package graph

import org.apache.spark.graphx.{Graph, Edge, VertexId}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}
import utils.{GalaxyException, MetaFile, FileHandler, FileLogger}

/**
 * Created by sara on 12/30/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object TriangleCounting {
  def main(args: Array[String]) {
    val filename = args(0)
    val output = args(1)
    val log = args(2)
    //val outputType = args(3);

    val start = System.currentTimeMillis()

    FileLogger.open(log)
    try {


      val conf = new SparkConf().setAppName("TriangleCounting")
      val sc = new SparkContext(conf)
      val sqlC = new SQLContext(sc);

      val metafiles = FileHandler.loadInput(filename)
      val uritype = metafiles(0).uriType

      val vertexFile = metafiles(0).uri
      val edgeFile = metafiles(1).uri

      val vertexRDD = sc.objectFile[(VertexId, Any)](vertexFile)
      val edgeRDD = sc.objectFile[Edge[Any]](edgeFile)

      val graph = Graph(vertexRDD, edgeRDD)

      val triCounts = graph.triangleCount().vertices.mapValues{v=> v.toDouble}

      val triangleFile = FileHandler.getFullName(sc, uritype, "tria")


      //triCounts.saveAsObjectFile(triangleFile)

      val schemaString = "vertex,count"
      val df = sqlC.createDataFrame(triCounts).toDF("vertex", "count")
      FileLogger.println("Scheme string: " + schemaString)
      df.write
        .format("com.databricks.spark.csv")
        .option("header", "true")
        .save(triangleFile)

      df.take(10).foreach(FileLogger.println(_))


      val objects = new Array[MetaFile](1)
      val verObj = new MetaFile("CSV[vertex,count]", uritype, triangleFile)
      objects.update(0, verObj)
      FileHandler.dumpOutput(output, objects)
      val stop = System.currentTimeMillis()

      FileLogger.println("Triangle counting successfully done in " + ((stop - start) / 1000.0 ) + " sec.")

    }catch {
      case e: Exception => FileLogger.println(GalaxyException.getString(e))
    } finally {
      FileLogger.close();
    }

  }

}
