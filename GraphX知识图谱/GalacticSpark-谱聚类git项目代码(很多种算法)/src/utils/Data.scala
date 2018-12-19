package utils

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vectors,Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SQLContext

/**
 * Created by sara on 8/19/15.
  * * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object Data {
  def loadData(sc:SparkContext, filepath:String) : RDD[LabeledPoint]  = {
    val data = sc.textFile(filepath);
    val parsedData = data.map { line =>
      val parts = line.split(',')
      try {
          LabeledPoint(parts.head.toDouble, Vectors.dense(parts.tail.map(_.toDouble)))
        } catch { case e: NumberFormatException => {null} }
      }
      parsedData.filter(a=>a!=null)
  }

  def loadData2Vector(sc:SparkContext, filepath:String) : RDD[Vector]  = {
    val data = sc.textFile(filepath);
    val parsedData = data.map { line =>
      val parts = line.split(' ')
      try {
        Vectors.dense(parts.map(_.toDouble))
      } catch { case e: NumberFormatException => {null} }
    }
    parsedData.filter(a=>a!=null)
  }

  def loadTSV2Vector(sc:SparkContext, filepath:String) : RDD[Vector]  = {
    val data = sc.textFile(filepath);
    val parsedData = data.map { line =>
      val parts = line.split(' ')
      try {
        Vectors.dense(parts.map(_.toDouble))
      } catch { case e: NumberFormatException => {null} }
    }
    parsedData.filter(a=>a!=null)
  }

  def loadTSV2Labeled(sc:SparkContext, filepath:String) : RDD[LabeledPoint]  = {
    val data = sc.textFile(filepath);
    val parsedData = data.map { line =>
      val parts = line.split(' ')
      try {
        LabeledPoint(parts.head.toDouble, Vectors.dense(parts.tail.map(_.toDouble)))
      } catch { case e: NumberFormatException => {null} }
    }
    parsedData.filter(a=>a!=null)
  }

  def loadCSV(sql:SQLContext, filepath: String, hasHeader: Boolean, inferSchema: Boolean) : DataFrame = {
    val df = sql.read
      .format("com.databricks.spark.csv")
      .option("header", hasHeader.toString) // Use first line of all files as header
      .option("inferSchema", inferSchema.toString)
      .load(filepath);
    df
  }


}
