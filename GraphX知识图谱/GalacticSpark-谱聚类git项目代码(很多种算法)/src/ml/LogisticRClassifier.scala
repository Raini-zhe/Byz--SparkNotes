package ml

import java.io.{File, PrintWriter}

//import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.util.MLReader

//import org.apache.spark.mllib.classification.LogisticRegressionModel
//import org.apache.spark.mllib.evaluation.MulticlassMetrics
//import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
//import org.apache.spark.mllib.evaluation.MulticlassMetrics
//import org.apache.spark.mllib.regression.LabeledPoint
//import org.apache.spark.mllib.linalg.Vectors
//import org.apache.spark.mllib.util.MLUtils
import utils.{FileHandler, Data}

/**
 * Created by sara on 8/20/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object LogisticRClassifier {
  def main(args: Array[String]) {

    val filename = args(0)
    val output = args(1);
    val log = args(2);
    val modelpath = args(3)
    val labeled = args(4).toBoolean

    try{
      val conf = new SparkConf().setAppName("LRClassifier")
      val sc = new SparkContext(conf)
      val sqlC = new SQLContext(sc)


      val metaFiles = FileHandler.loadInput(filename)
      val uriType = metaFiles(0).uriType
      val testData = Data.loadCSV(sqlC, metaFiles(0).uri,true,true)
      //val parsedData = sc.objectFile[LabeledPoint](metaFiles(0).uri)



      //val data = Data.loadData(sc, "file:/Users/sara/drp/spark/spark-1.4.1/data/kr_test.csv");
      //val data = Data.loadData(sc, args(0));

      //val modelpath = args(2)

      //val mlreader = new MLReader();


      val model = LogisticRegressionModel.load(sc, modelpath)
      

      // Compute raw scores on the test set.
//      if (labeled) {
//        val predictionAndLabels = data.map { case LabeledPoint(label, features) =>
//          val prediction = model.predict(features)
//          (prediction, label)
//        }
//
//        val pw = new PrintWriter(new File(args(1)));
//        for (a <- predictionAndLabels.collect()) {
//          pw.write(a._1 + " " + a._2 + "\n")
//        }
//
//        // Get evaluation metrics.
//        val metrics = new MulticlassMetrics(predictionAndLabels)
//        val precision = metrics.precision
//        pw.write("Precision = " + precision)
//
//        //************************
//        //the same as precisiion// Get evaluation metrics.
//        //val accuracy = 1.0 * predictionAndLabels.filter(x => x._1 == x._2).count() / data.count()
//
//        //pw.write("accuracy = " + accuracy)
//        //*********************
//
//        pw.close()
//
//
//      } else {
//        val predictionAndLabels = data.map { case LabeledPoint(label, features) =>
//          val prediction = model.predict(features)
//          (prediction)
//        }
//        val pw = new PrintWriter(new File(args(1)));
//        for (a <- predictionAndLabels.collect()) {
//          pw.write(a + "\n")
//        }
//        pw.close()
//      }
    }catch {
      case e: Exception => println("ERROR. tool unsuccessful:" + e);
    }
  }

}
