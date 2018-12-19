package inout

import java.io.{File, PrintWriter}

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}
import utils.{Data, FileHandler, FileLogger}


/**
 * Created by sara on 1/24/16.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object RDD2Data {
  def main(args: Array[String]) {
    val file = args(0)
    val output= args(1)
    val log = args(2)
    val inputType = args(3)
    val outputType = args(3)



    FileLogger.open(log)
    try {


      val conf = new SparkConf().setAppName("rdd2txt")
      val sc = new SparkContext(conf)
      val sqlC = new SQLContext(sc)

      val metaFiles = FileHandler.loadInput(file)
      val uriType = metaFiles(0).uriType
      val uri = metaFiles(0).uri

      if (inputType == "csv") {
        val df =  Data.loadCSV(sqlC, uri, true,false).cache()
        FileLogger.println("Points Count: " + df.count()  )
        if (outputType == "csv") {
          val pw = new PrintWriter(new File(output));
          pw.println(df.schema.fieldNames.mkString(","))
          val size = df.schema.fieldNames.length
          val fieldNames = df.schema.fieldNames
          val schema = df.schema
          val fieldIndex = fieldNames.map(s=>schema.fieldIndex(s))
          df.collect().foreach(r=> pw.println(fieldIndex.map(i=>r.getString(i)).mkString(",") ) )
          pw.close()
        }
      }

    } catch {
      case e: Exception => FileLogger.println("ERROR. tool unsuccessful:" + e);
    } finally {
      FileLogger.close()
    }
  }

}
