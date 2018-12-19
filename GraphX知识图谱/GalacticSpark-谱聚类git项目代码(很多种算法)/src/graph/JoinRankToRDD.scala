package graph

import java.io.{File, PrintWriter}

import org.apache.spark.{SparkConf, SparkContext}
import utils.{FileHandler, FileLogger, GalaxyException}

/**
  * Created by sara on 12/25/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
  */
object JoinRankToRDD {
   def main(args: Array[String]) {
     try {
       val filename = args(0);
       val output = args(1);
       val log = args(2);

       val other = args(3);

       val itemNum = args(4).toInt
       val order = args(5).toBoolean

       val conf = new SparkConf().setAppName("join")
       val sc = new SparkContext(conf)

       FileLogger.open(log);


       val metaFiles = FileHandler.loadInput(filename)
       val uritype = metaFiles(0).uriType
       val rankFile = metaFiles(0).uri

       val otherMetaFiles = FileHandler.loadInput(other)
       val otherUritype = otherMetaFiles(0).uriType
       val otherRddFile = otherMetaFiles(0).uri

       if (otherUritype != uritype) {
         FileLogger.error("Both rdds should use same file systems");
         throw GalaxyException;
       }

       val rank = sc.objectFile[(Long, Double)](rankFile)
       val rdd = sc.objectFile[(String, String)](otherRddFile).map{case (id, attr)=> (id.toLong, attr)};

       val joined = rank.join(rdd)
       val top = joined.sortBy(_._2._1, order).take(itemNum)

       val pw1 = new PrintWriter(new File(output));
       var i = 0;
       top.foreach{case (id, (value, name)) => i = i + 1; pw1.println(i.toString + " " + name + " " + value.toString)}
       pw1.close();

       FileLogger.println("Join rank successfull.");

     } catch {
       case e: Exception => FileLogger.println(GalaxyException.getString(e))
     } finally {
      FileLogger.close();
     }


   }
 }
