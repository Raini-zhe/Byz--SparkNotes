package graph;

/**
  * Created by sara on 11/4/15.
  * * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
  */
import java.io.{FileOutputStream, File, PrintWriter}

import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.types.{StructField, StructType,StringType}
import org.apache.spark.{SparkContext, SparkConf}
import utils._

import scala.io.Source
import org.apache.spark.graphx._


object DegreeGraph {
  def main(args:Array[String]) = {
    //each spark program has a spark conf
      val filename = args(0);
      val output = args(1);
      val log = args(2);

      val degreeType = args(3);
     // val outputType = args(4);
      FileLogger.open(log);

    try {

      val start = System.currentTimeMillis()

      val conf = new SparkConf().setAppName("degree")
      val sc = new SparkContext(conf)
      val sqlC = new SQLContext(sc);
      //    val filename = args(0);
      //    val linesIt =  Source.fromFile(filename).getLines();
      //    val lines = Array[String]("","","");
      //    linesIt.copyToArray(lines);
      //    val appId = lines(0).split("\t")(1);
      //
      //    val verAddr = lines(1).split("\t")(1);
      //    val verObj = sc.objectFile[(org.apache.spark.graphx.VertexId, Any)](verAddr);//deserialize
      //
      //    val edgeAddr = lines(2).split("\t")(1);
      //    val edgeObj = sc.objectFile[org.apache.spark.graphx.Edge[Int]](edgeAddr);


      val metaFiles = FileHandler.loadInput(filename)
      val uritype = metaFiles(0).uriType
      val vertexFile = metaFiles(0).uri
      val edgeFile = metaFiles(1).uri
      val vertexRDD = sc.objectFile[(VertexId, Any )](vertexFile)
      val edgeRDD = sc.objectFile[org.apache.spark.graphx.Edge[Any]](edgeFile);

      val graph = Graph(vertexRDD, edgeRDD);
      //graph.subgraph(vpred= (v, (at,r)) => r )
      val degree = if (degreeType == "in") { graph.inDegrees}
                    else if (degreeType =="out") {graph.outDegrees}
                    else {graph.degrees}

//      val outDegrees: VertexRDD[Int] = graph.outDegrees
//
//
//      val degreeGraph = graph.outerJoinVertices(outDegrees) { (id, oldAttr, outDegOpt) =>
//        outDegOpt match {
//          case Some(outDeg) => outDeg
//          case None => 0 // No outDegree means zero outDegree
//        }
//      }

      val degreeFile = FileHandler.getFullName(sc, uritype, "degree")

      //if (outputType == "csv") {

        val schemaString = "vertex,degree"
        val df = sqlC.createDataFrame(degree).toDF("vertex", "degree")
        FileLogger.println("Scheme string: " + schemaString)
        df.write
          .format("com.databricks.spark.csv")
          .option("header", "true")
          .save(degreeFile)
        df.take(10).foreach(FileLogger.println(_))
        //ToCSV.fromVertexRDD(sqlC, degree, degreeFile)


        val objects = new Array[MetaFile](1)
        val verObj = new MetaFile("CSV[vertex,degree]",uritype, degreeFile)

        objects.update(0, verObj)
        FileHandler.dumpOutput(output, objects)
     /* }
      else {
        degree.saveAsObjectFile(degreeFile);
        val objects = new Array[MetaFile](1)
        val verObj = new MetaFile("RDD[(long,long)]",uritype, degreeFile)

        objects.update(0, verObj)
        FileHandler.dumpOutput(output, objects)
      }
    */
      val stop = System.currentTimeMillis()

      FileLogger.println("Degree computation successfully done in " + ((stop - start)/1000.0) + " sec.");
    } catch {
      case e: Exception => FileLogger.println(GalaxyException.getString(e))
    } finally {
      FileLogger.close();
    }
  }

}
