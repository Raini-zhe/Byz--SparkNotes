package ml

import java.io.{File, PrintWriter}

import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.regression._
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}




import utils.{MetaFile, FileHandler, FileLogger, Data}

/**
 * Created by sara on 8/20/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object LinearRegressionTool {
  def main(args: Array[String]) {
    val filename = args(0)
    val output = args(1);
    val log = args(2);
    val modelpath = args(3)
    val modelType = args(4).toString



    try
    {
      val conf = new SparkConf().setAppName("lr")
      val sc = new SparkContext(conf)
      val sqlC = new SQLContext(sc)

      val metaFiles = FileHandler.loadInput(filename)
      val uriType = metaFiles(0).uriType
      val parsedData = sc.objectFile[LabeledPoint](metaFiles(0).uri)


      val model = if (modelType == "LinearRegressionWithSGD") {
        //print("LinearRegressionWithSGD")
        LinearRegressionModel.load(sc, modelpath)
      }else if (modelType == "RidgeRegressionWithSGD") {
        //print("RidgeRegressionWithSGD")
        RidgeRegressionModel.load(sc,modelpath)
      } else{
        //print("Lasso")
        LassoModel.load(sc, modelpath)
      }
      //val model = LinearRegressionModel.load(sc, modelpath)

      // Evaluate model on training examples and compute training error
      val valuesAndPreds = parsedData.map { point =>
          val prediction = model.predict(point.features)
          (point.label, prediction)
      }





      val metrics = new RegressionMetrics(valuesAndPreds)
      FileLogger.println("Number of points:" + parsedData.count())
      FileLogger.println("Note: if the input data is unlabeled then the following metric is not meaningful.")
      // Squared error
      FileLogger.println(s"MSE = ${metrics.meanSquaredError}")
      FileLogger.println(s"RMSE = ${metrics.rootMeanSquaredError}")

      // R-squared
      FileLogger.println(s"R-squared = ${metrics.r2}")

      // Mean absolute error
      FileLogger.println(s"MAE = ${metrics.meanAbsoluteError}")

      // Explained variance
      FileLogger.println(s"Explained variance = ${metrics.explainedVariance}")

      val fullname = FileHandler.getFullName(sc, uriType, "linear-reg-output")
      val df = sqlC.createDataFrame(valuesAndPreds).toDF("output", "pred")
        .write
        .format("com.databricks.spark.csv")
        .option("header", "true")
        .save(fullname)


      val metaFile = new MetaFile("CSV[output,pred]", uriType, fullname)
      FileHandler.dumpOutput(output, Array(metaFile))



//        val pw = new PrintWriter(new File(args(1)));
//        for (a <- valuesAndPreds.collect()) {
//          pw.write(a._1 + " " + a._2 + "\n")
//        }
//
//        val MSE = valuesAndPreds.map { case (v, p) => math.pow((v - p), 2) }.mean()
//        pw.write("training Mean Squared Error = " + MSE)
//        pw.close()
//
//      } else{
//        val valuesAndPreds = data.map { point =>
//          val prediction = model.predict(point.features)
//          ( prediction)
//        }
//        val pw = new PrintWriter(new File(args(1)));
//        for (a <- valuesAndPreds.collect()) {
//          pw.write(a + "\n")
//        }
//        pw.close()
//      }

    }catch{
      case e: Exception => FileLogger.println("ERROR. tool unsuccessful:" + e);
    } finally {
      FileLogger.close();
    }
  }

}
