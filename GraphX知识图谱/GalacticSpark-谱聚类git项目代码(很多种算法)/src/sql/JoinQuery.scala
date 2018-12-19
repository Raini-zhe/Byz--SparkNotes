package sql

import java.io.{File, PrintWriter}

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import utils._
import scala.io._
/**
  * Created by sara on 12/25/15.
  * * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
  */
object JoinQuery {
   def main(args: Array[String]) {

     val file1 = args(0)
     val output = args(1)
     val log = args(2)
     val file2 = args(3)
     val table1 = args(4)
     val table2 = args(5)
     val sqlFile = args(6)
     FileLogger.open(log);
     try {
       val start = System.currentTimeMillis();

       val conf = new SparkConf().setAppName("joinQuery")
       val sc = new SparkContext(conf)
       val sqlC = new SQLContext(sc);


       val sqlLines =  Source.fromFile(sqlFile).getLines()
       val resultSchema = sqlLines.next().split(",").map(s=>s.stripMargin)
       val sqlRaw= sqlLines.next();
       // val inputType = args(5)
       val sql = sqlRaw.replace("__gt__", ">").replace("__lt__", "<").replace("__sq__", "'").replace("__st__", "*")




       val metaFiles = FileHandler.loadInput(file1)
       val uriType = metaFiles(0).uriType
       val uri = metaFiles(0).uri

       val otherMetaFiles = FileHandler.loadInput(file2)
       val otherUriType = otherMetaFiles(0).uriType
       val otherURI = otherMetaFiles(0).uri

       if (otherUriType != uriType) {
         FileLogger.error("Both files should use the same file systems");
         throw GalaxyException;
       }

       val df1 =  Data.loadCSV(sqlC, uri, true, true)
       val df2 =  Data.loadCSV(sqlC, otherURI, true, true)
       df1.registerTempTable(table1);
       df2.registerTempTable(table2);
       FileLogger.println("Query: " + sql)
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

       FileLogger.println("Results schema: " + resultsDF.schema.fieldNames.mkString(","))
       FileLogger.println("Results count: " + results.count())

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

       val stop = System.currentTimeMillis()

       FileLogger.println("Join query successfully done in " + ((stop-start) / 1000.0) + " sec");

     } catch {
       case e: Exception => FileLogger.println(GalaxyException.getString(e))
     } finally {
      FileLogger.close();
     }


   }
 }
