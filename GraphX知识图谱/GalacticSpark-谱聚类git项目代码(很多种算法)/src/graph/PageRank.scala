package graph

import org.apache.spark.graphx._
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}
import utils.{GalaxyException, MetaFile, FileHandler, FileLogger}

//import scala.reflect.runtime.{ universe => ru }
/**
 * Created by sara on 12/25/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object PageRank {

//  def getType[T](clazz: Class[T]):ru.Type = {
//    val runtimeMirror =  ru.runtimeMirror(clazz.getClassLoader)
//    runtimeMirror.classSymbol(clazz).toType
//  }
  def main(args: Array[String]) {

      val filename = args(0);
      val output = args(1);
      val log = args(2);

      FileLogger.open(log);
      val thr = args(3).toDouble
      //val outputType = args(4);
      //val outputType = "csv";

    try {
      val start = System.currentTimeMillis()

      val conf = new SparkConf().setAppName("pagerank");
      val sc = new SparkContext(conf)
      val sqlC = new SQLContext(sc);

      val metaFiles = FileHandler.loadInput(filename)
      val uritype = metaFiles(0).uriType
      val vertexFile = metaFiles(0).uri
      val edgeFile = metaFiles(1).uri
      val vertexRDD = sc.objectFile[(VertexId, Any )](vertexFile)
      val edgeRDD = sc.objectFile[org.apache.spark.graphx.Edge[Any]](edgeFile);

      val graph = Graph(vertexRDD, edgeRDD)

      //val graph = GraphLoader.edgeListFile(sc, metaFiles(0).uri)




      val ranks = graph.pageRank(thr).vertices//.mapValues(d=>d.toString)

      val rankFile = FileHandler.getFullName(sc, uritype, "rank")

      //if (outputType == "csv") {

        val schemaString = "vertex, rank"
        val df = sqlC.createDataFrame(ranks).toDF("vertex", "rank")
        FileLogger.println("Scheme string: " + schemaString)
        df.write
          .format("com.databricks.spark.csv")
          .option("header", "true")
          .save(rankFile)
        df.take(10).foreach(FileLogger.println(_))
        //ToCSV.fromVertexRDD(sqlC, degree, degreeFile)

        val objects = new Array[MetaFile](1)
        val verObj = new MetaFile
        verObj.objName = "CSV[vertex,rank]"
        verObj.uri = rankFile
        verObj.uriType = uritype

        objects.update(0, verObj)
        FileHandler.dumpOutput(output, objects)
      /*}
      else {
        ranks.saveAsObjectFile(rankFile)
        val objects = new Array[MetaFile](1)
        val verObj = new MetaFile
        verObj.objName = "RDD[(long,double)]"
        verObj.uri = rankFile
        verObj.uriType = uritype

        objects.update(0, verObj)
        FileHandler.dumpOutput(output, objects)

      }*/

      val stop = System.currentTimeMillis();
      FileLogger.println("Pagerank successfully done in " + ((stop - start) / 1000.0) + " sec" );


    } catch {
      case e: Exception => FileLogger.println(GalaxyException.getString(e))
    } finally {
      FileLogger.close();
    }
  }

}
