package ml

import java.io.{File, PrintWriter}

import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.util.MLUtils

/**
 * Created by sara on 7/12/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */

object DTClassifier {


  def main(args: Array[String]) {
    try {

      val conf = new SparkConf().setAppName("kmean").setMaster("local")
      val sc = new SparkContext(conf)

      // Load and parse the data file.
      val filename = args(0);
      val data = MLUtils.loadLibSVMFile(sc, "file:" + filename)
      // Split the data into training and test sets (30% held out for testing)
      val splits = data.randomSplit(Array(0.7, 0.3))
      val (trainingData, testData) = (splits(0), splits(1))

      // Train a DecisionTree model.
      //  Empty categoricalFeaturesInfo indicates all features are continuous.
      /*val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32*/

      //val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      // impurity, maxDepth, maxBins)
      val modelPath = args(2)
      val model = DecisionTreeModel.load(sc, modelPath)


      // Evaluate model on test instances and compute test error
      val labelAndPreds = testData.map { point =>
        val prediction = model.predict(point.features)
        (point.label, prediction)
      };

      val pw = new PrintWriter(new File(args(1)));
      for( a <- labelAndPreds.collect()) {
        pw.write(a._1 + " " + a._2 + "\n")
      }

      val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
      pw.write("Test Error = " + testErr)
      //println("Learned classification tree model:\n" + model.toDebugString)
      pw.close()
    }catch {
      case e: Exception => println("ERROR. tool unsuccessful:" + e);
    }
    }


  }
