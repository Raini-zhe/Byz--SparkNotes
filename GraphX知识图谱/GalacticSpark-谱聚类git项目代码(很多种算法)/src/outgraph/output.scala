
package outgraph

/**
	* Created by sara on 8/18/15.
	* Copyright (C) 2015 by Sara Riazi
	* University of Oregon
	* All rights reserved.
	*/

import java.io.{FileOutputStream, File, PrintWriter}

import org.apache.spark.{SparkContext, SparkConf}

import scala.io.Source
import org.apache.spark.graphx._


object OutputGraph {
	def main(args:Array[String]) = {
		val conf = new SparkConf().setAppName("output").setMaster("local")
		val sc = new SparkContext(conf)

		val filename = args(0);
		val linesIt =  Source.fromFile(filename).getLines();
		val lines = Array[String]("","","");
		linesIt.copyToArray(lines);
		val appId = lines(0).split("\t")(1);

		val verAddr = lines(1).split("\t")(1);
		val verObj = sc.objectFile[(org.apache.spark.graphx.VertexId, Any)](verAddr);

		val edgeAddr = lines(2).split("\t")(1);
		val edgeObj = sc.objectFile[org.apache.spark.graphx.Edge[Int]](edgeAddr);

		val verAr = verObj.collect();
		val pw = new PrintWriter(new FileOutputStream(args(1), true));
		for(i <- 0 until verAr.length){
			pw.println(verAr(i)._1.toLong + " " + verAr(i)._2.toString());
		}
		pw.close();
		//verObj.saveAsTextFile(args(1));
				
	}
}
