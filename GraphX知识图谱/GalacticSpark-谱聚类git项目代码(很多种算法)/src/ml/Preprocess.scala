package ml

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}
import utils._
import scala.io._

/**
  * Created by sara on 1/22/16.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
  */
object Preprocess {
  def main(args: Array[String]) {
    val datafile = args(0)
    val output = args(1);
    val log = args(2);
    val featurefile = args(3)
    val labelName = args(4)

    val conf = new SparkConf().setAppName("linearLearn")
    val sc = new SparkContext(conf)
    val sqlC = new SQLContext(sc)

    val metaFiles = FileHandler.loadInput(datafile)
    val uritype = metaFiles(0).uriType
    val lines = Source.fromFile(featurefile).getLines()

    val dataDF = Data.loadCSV(sqlC, datafile, true, true)



  }

}
