package ml

import java.io.{File, PrintWriter}

import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}
import utils._

/**
 * Created by sara on 8/19/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object SVMsClassifier {
  def main (args: Array[String]) {
    val filename = args(0)
    val output = args(1);
    val log = args(2);
    val modelpath = args(3)
    FileLogger.open(log)

    try {

      val conf = new SparkConf().setAppName("svmClassifier")
      val sc = new SparkContext(conf)
      val sqlC = new SQLContext(sc)
      //val labeled = args(3).toBoolean

      //val data = Data.loadData(sc, "file:/Users/sara/drp/data/kr_test.csv");
      //val filename = args(0)
      //val objects = FileHandler.loadInput(filename);
      //FileLogger.open(args(2))
      //val uritype = objects(0).uriType
      //val training = Data.loadData(sc, objects(0).uri)
      //val data = sc.objectFile[LabeledPoint](objects(0).uri)
      val metaFiles = FileHandler.loadInput(filename)
      val uriType = metaFiles(0).uriType
      val parsedData = sc.objectFile[LabeledPoint](metaFiles(0).uri)

      //val data = Data.loadData(sc, args(0));

      //val model = SVMModel.load(sc, "myModelPath")
      val model = SVMModel.load(sc, modelpath)

      val scoreAndLabels = parsedData.map { point =>
        val score = model.predict(point.features)
        (score, point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)

      val auROC = metrics.areaUnderROC()
      

      val pr = metrics.areaUnderPR()
      //FileLogger.println("Accuracy = " + accuracy);
      FileLogger.println("Area under ROC = " + auROC)
      FileLogger.println("Area under pr = " + pr)
      // Compute raw scores on the test set.


      val fullname = FileHandler.getFullName(sc, uriType, "svm-output")
      val df = sqlC.createDataFrame(scoreAndLabels).toDF("pred", "label")
        .write
        .format("com.databricks.spark.csv")
        .option("header", "true")
        .save(fullname)

      val metaFile = new MetaFile("RDD[(output,predication)]", uriType, fullname)
      FileHandler.dumpOutput(output, Array(metaFile))
      /*
      if (labeled) {
        val scoreAndLabels = data.map { point =>
          val score = if  (model.predict(point.features) >= 0.0) 1.0 else 0.0
          (score, point.label)

        }
        val error = scoreAndLabels.map(a => if (a._1 == a._2) 1.0 else 0.0)
        val sum = error.fold(0.0)((s,p) => s + p)
        val accuracy = sum / error.count()

        val pw = new PrintWriter(new File(args(1)));
        for (a <- scoreAndLabels.collect()) {
          pw.write(a._1 + " " + a._2 + "\n")
        }
        pw.write("accuracy = " + accuracy)
        pw.close()

      } else {
        val scoreAndLabels = data.map { point =>
          val score = model.predict(point.features)
          (score)
        }
        val pw = new PrintWriter(new File(args(1)));
        for (a <- scoreAndLabels.collect()) {
          pw.write(a + "\n")
        }
        pw.close()
      }
      */
    }catch {
      case e: Exception => FileLogger.println(GalaxyException.getString(e));
    } finally {
      FileLogger.close()
    }
  }

}
