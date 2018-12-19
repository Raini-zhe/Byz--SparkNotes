package graph

import java.io.{File, PrintWriter}

import org.apache.spark.graphx._
import org.apache.spark.{SparkContext, SparkConf}
import utils.{GalaxyException, FileHandler, FileLogger}

/**
 * Created by sara on 11/4/15.
  * * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object ClusterCompare {

  def main(args: Array[String]) {
    try {
      val filename = args(0)
      val output = args(1)
      val metric = args(2)
      FileLogger.open(output)
      val objects = FileHandler.loadInput(filename);
      val vertexFile = objects(0).uri
      val edgeFile = objects(1).uri

      val conf = new SparkConf().setAppName("clusterEval");
      conf.set("spark.scheduler.mode", "FAIR")

      val sc = new SparkContext(conf)
      val vertexRDD = sc.objectFile[(VertexId, Int)](vertexFile)
      val edgeRDD = sc.objectFile[org.apache.spark.graphx.Edge[Int]](edgeFile);
      val g = Graph(vertexRDD, edgeRDD)

//      val pw = new PrintWriter(new File(output));


      if (metric.equals("ncut")) {
        val start = System.currentTimeMillis();
        val ncut = new NCut().evaluate(g);
        val stop = System.currentTimeMillis();
        FileLogger.println("ncut = " + ncut)
        FileLogger.println("ncut computation time: " + (stop- start)/ 1000.0 + "sec" )

      } else if (metric.equals("modularity")) {
        val modularity = new Modularity().evaluate(g)
        FileLogger.println("modularity = " + modularity)
      } else {
        val start = System.currentTimeMillis();

        val modularity = new Modularity().evaluate(g)
        val stop = System.currentTimeMillis();

        val ncut = new NCut().evaluate(g);
        val stop2 = System.currentTimeMillis();

        FileLogger.println("ncut = " + ncut)
        FileLogger.println("modularity = " + modularity)
        FileLogger.println("ncut computation time: " + (stop2 - stop)/ 1000.0 + "sec" )
        FileLogger.println("modularity computation time: " + (stop - start)/ 1000.0 + "sec" )

      }
      FileLogger.close();

    } catch {
      case e: Exception => println(GalaxyException.getString(e))
    }
  }
}
