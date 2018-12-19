package utils

import org.apache.spark.graphx.{VertexRDD, VertexId}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext

/**
 * Created by sara on 1/2/16.
  * * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object ToCSV {

  def fromVertexRDD(sql: SQLContext, rdd : VertexRDD[Any], path: String) : Unit = {
//
//    val newR = rdd match {
//      case r: RDD[(VertexId, (Int, Int))] => r.map{case (a,(b1,b2))=>(a,b1,b2)}
//      case r: RDD[(VertexId, (String, Int))] => r.map{case (a,(b1,b2))=>(a,b1,b2)}
//      case r: RDD[(VertexId, (Int, String))] => r.map{case (a,(b1,b2))=>(a,b1,b2)}
//      case r: RDD[(VertexId, (Double, Int))] => r.map{case (a,(b1,b2))=>(a,b1,b2)}
//      case r: RDD[(VertexId, (Int, Double))] => r.map{case (a,(b1,b2))=>(a,b1,b2)}
//      case r: RDD[(VertexId, (String, Double))] => r.map{case (a,(b1,b2))=>(a,b1,b2)}
//      case r: RDD[(VertexId, (Double, String))] => r.map{case (a,(b1,b2))=>(a,b1,b2)}
//      case r: RDD[(VertexId, (Double, Double))] => r.map{case (a,(b1,b2))=>(a,b1,b2)}
//      //case  => rdd
//    }
//
//    val df = sql.createDataFrame(newR);
//    df.write
//      .format("com.databricks.spark.csv")
//      .save(path)
  }

}
