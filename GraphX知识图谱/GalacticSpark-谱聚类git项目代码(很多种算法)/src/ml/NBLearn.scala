package ml

import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.mllib.classification.NaiveBayes

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import utils._


/**
 * Created by sara on 7/30/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object NBLearn {
  def main(args: Array[String]) {
    try {

      val conf = new SparkConf().setAppName("kmean").setMaster("local")
      val sc = new SparkContext(conf)

      val filename = args(0);
      FileLogger.open(args(2))
      //val data = sc.textFile("file:/Users/sara/drp/spark/spark-1.4.1/data/mllib/sample_naive_bayes_data.txt")
      val objects =FileHandler.loadInput(filename)
      //val data = sc.textFile(objects(0).uri)
//
//      val parsedData = data.map { line =>
//        val vars = line.split(',')
//        val label = vars.head
//        val x = vars.tail
//        LabeledPoint(label.toDouble, Vectors.dense(x.map(_.toDouble)))
//      }

      val parsedData = Data.loadData(sc, objects(0).uri)
      // Split data into training (60%) and test (40%).
      //val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
      //val training = splits(0)
      //val test = splits(1)
      val lambda = args(4).toFloat
      val modelType = args(3)
      val training = parsedData
      //val nb = new NaiveBayes()
      val paramMaps = new ParamGridBuilder()
   //   paramMaps.addGrid(nb.smoothing)

      val model = NaiveBayes.train(training, lambda , modelType)


      val predictionAndLabel = training.map(p => (model.predict(p.features), p.label))
      val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / training.count()
      FileLogger.println("Training accuracy = " + accuracy)

      // Save and load model
      //model.save(sc, "/Users/sara/drp/myNBModelPath")
      model.save(sc , "file:" + args(1))
      //val sameModel = NaiveBayesModel.load(sc, "/Users/sara/drp/myNBModelPath")

    } catch {
      case e: Exception => println(GalaxyException.getString(e));
    }

  }
}