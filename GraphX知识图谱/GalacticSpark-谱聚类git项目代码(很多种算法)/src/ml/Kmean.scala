package ml

import java.io.{File, FileOutputStream, PrintWriter}

import org.apache.spark.graphx.GraphLoader
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import utils.{GalaxyException, FileHandler}

/**
 * Created by sara on 7/2/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object Kmean {



  def main (args: Array[String]) {
    try {
    val conf = new SparkConf().setAppName("kmean")
    val sc = new SparkContext(conf)

    val filename = args(0);
    val metaFiles = FileHandler.loadInput(filename)
    println("metaFiles = " + metaFiles(0).uri)
    val uritype = metaFiles(0).uriType

    val data = sc.textFile(metaFiles(0).uri)
    println("parts = " + data.partitions.length)

    // Load and parse the data
    //val data = sc.textFile("file:/Users/sara/drp/spark/spark-1.4.1/data/mllib/kmeans_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()

    // Cluster the data into two classes using KMeans
    val numClusters = args(3).toInt
    val numIterations = args(4).toInt
    val clusters = KMeans.train(parsedData, numClusters, numIterations)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(parsedData)
    //clusters.clusterCenters.foreach( v=>  println(v))
    val pw2 = new PrintWriter(new File(args(2)));
    pw2.println("Within Set Sum of Squared Errors")
    pw2.println("WSSSE = " + WSSSE)
    pw2.println("Normalized WSSSE = " + WSSSE/parsedData.count());
    pw2.close();
    // Save and load model
      clusters.save(sc, "file:" + args(1));
    } catch {
      case e: Exception => println(GalaxyException.getString(e))

    }

     // println("save successful");
   // val sameModel = KMeansModel.load(sc, args(1))

   // val pw = new PrintWriter(new FileOutputStream( args(1), true));
   // pw.write("Cluster centers:\n")
   // clusters.clusterCenters.foreach( v=> pw.write( (v.toArray.map(e=>e.toString)).mkString(",") + "\n"))
   // pw.close();

  }
}
