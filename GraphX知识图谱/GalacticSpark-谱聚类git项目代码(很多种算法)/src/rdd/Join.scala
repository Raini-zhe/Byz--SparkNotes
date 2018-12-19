package rdd

import org.apache.spark.{SparkContext, SparkConf}
import utils.{FileHandler, GalaxyException, FileLogger}

/**
 * Created by sara on 12/25/15.
  ** Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object Join {
  def main(args: Array[String]) {
    try {
      val filename = args(0);
      val output = args(1);
      val log = args(2);

      val other = args(3);

      val conf = new SparkConf().setAppName("join")
      val sc = new SparkContext(conf)


      val metaFiles = FileHandler.loadInput(filename)
      val uritype = metaFiles(0).uriType
      val rddFile = metaFiles(0).uri

      val otherMetaFiles = FileHandler.loadInput(other)
      val otherUritype = otherMetaFiles(0).uriType
      val otherRddFile = otherMetaFiles(0).uri

      if (otherUritype != uritype) {
        FileLogger.error("Both rdds should use same file systems");
        throw GalaxyException;
      }

      val rdd = sc.objectFile[(Long, String)](rddFile)
      val otherRdd = sc.objectFile[(Long, String)](otherRddFile);

      val joined = rdd.join(otherRdd)
      joined.mapValues{case (s1, s2)=>s1}

      FileLogger.println("Pagerank successfull.");
      FileLogger.close();
    } catch {
      case e: Exception => println(GalaxyException.getString(e))
    }

  }
}
