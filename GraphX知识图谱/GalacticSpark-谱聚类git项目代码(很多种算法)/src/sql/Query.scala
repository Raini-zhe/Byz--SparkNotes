package sql

import java.io.{File, FileReader}

import org.apache.spark.graphx.VertexId
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkContext, SparkConf}
import utils._
import scala.io.Source


/**
 * Created by sara on 12/29/15.
  * * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object Query {
  def main(args: Array[String]) {
    val filename = args(0)
    val output = args(1)
    val log = args(2)
    val table = args(3).stripMargin
    val sqlFile = args(4)


    FileLogger.open(log);
    try {


      val sqlLines =  Source.fromFile(sqlFile).getLines()
      val resultSchema = sqlLines.next().split(",").map(s=>s.stripMargin)
      val sqlRaw= sqlLines.next();


     // val inputType = args(5)
      val sql = sqlRaw.replace("__gt__", ">").replace("__lt__", "<").replace("__sq__", "'").replace("__st__", "*")
      val conf = new SparkConf().setAppName("query")
      val sc = new SparkContext(conf)
      val sqlC = new SQLContext(sc)


      val metaFiles = FileHandler.loadInput(filename)
      val uriType = metaFiles(0).uriType
      val uri = metaFiles(0).uri

      FileLogger.println("Query:" + sql)

      //sqlContext.createDataFrame()
      //val df = if (inputType == "csv") {

      val df =  Data.loadCSV(sqlC, uri, true, true)
      //} else {
      //  val rdd = sc.objectFile[(Int,Int)](uri);
      //  sqlContext.createDataFrame(rdd)
      //}


      df.registerTempTable(table);
      //FileLogger.println("table name: " + table);

      val results = sqlC.sql(sql)
      val resultsDF = resultSchema.length match {
        case 1 => results.toDF(resultSchema(0))
        case 2 => results.toDF(resultSchema(0), resultSchema(1))
        case 3 => results.toDF(resultSchema(0), resultSchema(1), resultSchema(2))
        case 4 => results.toDF(resultSchema(0), resultSchema(1), resultSchema(2), resultSchema(3))
        case 5 => results.toDF(resultSchema(0), resultSchema(1), resultSchema(2), resultSchema(3), resultSchema(4))
        case 6 => results.toDF(resultSchema(0), resultSchema(1), resultSchema(2), resultSchema(3), resultSchema(4), resultSchema(6))
        case _ => results.toDF()
      }
      FileLogger.println("Results count: " + results.count())
      FileLogger.println("Results schema: " + resultsDF.schema.fieldNames.mkString(","))

      results.take(10).foreach{r=> FileLogger.println(r.toString())}
      //FileLogger.println("Results table:")

      //results.collect().foreach { case row => FileLogger.println(row.mkString(",")) }
      val resultsFile = FileHandler.getFullName(sc, uriType, "queryresult")
      results.write
        .format("com.databricks.spark.csv")
        .option("header", "true")
        .save(resultsFile)
      val obj = new MetaFile("CSV[" + resultsDF.schema.fieldNames.mkString(",")+"]",uriType, resultsFile)
      FileHandler.dumpOutput(output, Array(obj))
    }catch {
      case e: Exception => FileLogger.println(GalaxyException.getString(e))
    } finally {
      FileLogger.close();
    }

  }

}
