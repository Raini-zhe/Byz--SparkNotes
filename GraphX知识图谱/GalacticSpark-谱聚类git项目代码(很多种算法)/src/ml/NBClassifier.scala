package ml

import java.io.{PrintWriter, File}

import org.apache.spark.mllib.classification.{NaiveBayesModel, NaiveBayes}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext, SparkConf}


/**
 * Created by sara on 7/31/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object NBClassifier {

  def main(args: Array[String]) {
    try {

      val conf = new SparkConf().setAppName("kmean").setMaster("local")
      val sc = new SparkContext(conf)

      val filename = args(0);
      //val data = sc.textFile("file:/Users/sara/drp/spark/spark-1.4.1/data/mllib/sample_naive_bayes_data.txt")
      val data = sc.textFile("file:" + filename)
      val parsedData = data.map { line =>
        val parts = line.split(',')
        LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
      }
      // Split data into training (60%) and test (40%).
      val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
      val training = splits(0)
      val test = splits(1)

      //val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")

      val modelPath = args(2)
      val model = NaiveBayesModel.load(sc, modelPath)

      val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
      val pw = new PrintWriter(new File(args(1)));
      for (a <- predictionAndLabel.collect()) {
        pw.write(a._1 + " " + a._2 + "\n")
      }




      val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

      pw.write("accuracy = " + accuracy)
      pw.close()

      // Save and load model
      //model.save(sc, "/Users/sara/drp/myNBModelPath")
      //model.save(sc , "file:" + args(1))


    }catch {
      case e: Exception => println("ERROR. tool unsuccessful:" + e);
    }

  }
}
