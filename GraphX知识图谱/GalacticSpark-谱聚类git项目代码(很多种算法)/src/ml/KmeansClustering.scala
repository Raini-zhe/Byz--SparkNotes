package ml

import java.io.{File, FileOutputStream, PrintWriter}

import org.apache.spark.graphx.GraphLoader
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import utils.{FileLogger, GalaxyException, FileHandler}
import org.apache.spark.mllib.linalg.{Vectors,Vector}
/**
 * Created by sara on 7/2/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object KmeansClustering {



  def main (args: Array[String]) {
    try {
      val conf = new SparkConf().setAppName("kmean")
      val sc = new SparkContext(conf)

      val filename = args(0);
      val output = args(1)
      val log = args(2)
      FileLogger.open(log);

      val metaFiles = FileHandler.loadInput(filename)
      val uritype = metaFiles(0).uriType
      val data = sc.objectFile[Vector](metaFiles(0).uri)
      // Load and parse the data
      //val data = sc.textFile("file:/Users/sara/drp/spark/spark-1.4.1/data/mllib/kmeans_data.txt")
      //val parsedData = data.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()

      // Cluster the data into two classes using KMeans
      val numClusters = args(3).toInt
      val numIterations = args(4).toInt
      val clusters = KMeans.train(data, numClusters, numIterations)

      // Evaluate clustering by computing Within Set Sum of Squared Errors
      val WSSSE = clusters.computeCost(data)
      //clusters.clusterCenters.foreach( v=>  println(v))
      FileLogger.println("Within Set Sum of Squared Errors")
      FileLogger.println("WSSSE = " + WSSSE)
      FileLogger.println("Normalized WSSSE = " + WSSSE/data.count());
      // Save and load model
      clusters.save(sc, output);
    } catch {
      case e: Exception => FileLogger.println(GalaxyException.getString(e))
    } finally {
      FileLogger.close();
    }

     // println("save successful");
   // val sameModel = KMeansModel.load(sc, args(1))

   // val pw = new PrintWriter(new FileOutputStream( args(1), true));
   // pw.write("Cluster centers:\n")
   // clusters.clusterCenters.foreach( v=> pw.write( (v.toArray.map(e=>e.toString)).mkString(",") + "\n"))
   // pw.close();

  }
}
