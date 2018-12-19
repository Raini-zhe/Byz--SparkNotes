package ml

import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.linalg.{DenseVector, Vectors, Vector}
import utils.{FileLogger, FileHandler, Data}

/**
 * Created by sara on 8/20/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object LinearRLearn {
  def main(args: Array[String]) {

    val filename = args(0)
    val output = args(1);
    val log = args(2);

    val modelType = args(3).toString
    val numIterations = args(4).toInt
    FileLogger.open(log)
    try {
      val start = System.currentTimeMillis()
      //val classColName = args(3)

      val conf = new SparkConf().setAppName("linearLearn")
      val sc = new SparkContext(conf)
      //val sqlC = new SQLContext(sc)

      val metaFiles = FileHandler.loadInput(filename)
      val uritype = metaFiles(0).uriType
      val parsedData = sc.objectFile[LabeledPoint](metaFiles(0).uri).cache()
      // Load and parse the data

      //val parsedData = Data.loadData(sc, "file:" + filename)
      /*val dataDF = Data.loadCSV(sqlC, metaFiles(0).uri, false)

      val labelIndex = dataDF.columns.indexOf(classColName)

      val parsedData = dataDF.rdd.map{r=>val f = r.toSeq.toArray.zipWithIndex.filter{case (c,i)=>i== labelIndex}.map{case (c,i)=>c.asInstanceOf[Double]}
          LabeledPoint(r.getDouble(labelIndex), Vectors.dense(f))
      }
      */
      //dataDF.map{ r =>
      //  val index = r.
      //    r.fieldIndex(classColName)
      //  Vectors.dense()
      //  LabeledPoint(r.getDouble(index), r.)

      //       }

      /*
      val data = sc.textFile("file:/Users/sara/drp/spark/spark-1.4.1/data/mllib/ridge-data/lpsa.data")
      val parsedData = data.map { line =>
        val parts = line.split(',')
        LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
      }.cache()
*/
      // Building the model

      //val numIterations = 100

      val model = if (modelType == "LinearRegressionWithSGD") {
        //print("LinearRegressionWithSGD")
         LinearRegressionWithSGD.train(parsedData, numIterations)
      }else if (modelType == "RidgeRegressionWithSGD") {
        //print("RidgeRegressionWithSGD")
         RidgeRegressionWithSGD.train(parsedData, numIterations)
      } else{
        //print("Lasso")
         LassoWithSGD.train(parsedData, numIterations)
      }
      // Evaluate model on training examples and compute training error
      val valuesAndPreds = parsedData.map { point =>
        val prediction = model.predict(point.features)
        (point.label, prediction)
      }

      //val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
      //FileLogger.println("training Mean Squared Error = " + MSE)

      // Save and load model
      //model.save(sc, "myModelPath")
      model.save(sc, output)
      //val sameModel = LinearRegressionModel.load(sc, "myModelPath")
     // val sameModel = RidgeRegressionModel.load(sc,"myModelPath")
      //val sameModel = LassoModel.load(sc, "myModelPath")

      val metrics = new RegressionMetrics(valuesAndPreds)

      // Squared error
      FileLogger.println(s"MSE = ${metrics.meanSquaredError}")
      FileLogger.println(s"RMSE = ${metrics.rootMeanSquaredError}")

      // R-squared
      FileLogger.println(s"R-squared = ${metrics.r2}")

      // Mean absolute error
      FileLogger.println(s"MAE = ${metrics.meanAbsoluteError}")

      // Explained variance
      FileLogger.println(s"Explained variance = ${metrics.explainedVariance}")

      val stop = System.currentTimeMillis()

      FileLogger.println("Linear Regression successfully done in " + ((stop - start) / 1000.0) + " sec" )

    }catch {
      case e: Exception => FileLogger.error( "ERROR. tool unsuccessful:" + e );
    } finally {
      FileLogger.close()
    }

  }

}
