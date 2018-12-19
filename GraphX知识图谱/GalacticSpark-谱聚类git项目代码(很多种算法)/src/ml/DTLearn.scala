package ml

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by sara on 7/12/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object DTLearn {


  def main (args: Array[String]) {
    try {

    val conf = new SparkConf().setAppName("kmean").setMaster("local")
    val sc = new SparkContext(conf)

    // Load and parse the data file.
    val filename = args(0);
    val data = MLUtils.loadLibSVMFile(sc, "file:" + filename)
    //val data = MLUtils.loadLibSVMFile(sc, "file:/Users/sara/drp/spark/spark-1.4.1/data/mllib/sample_libsvm_data.txt")
    // Split the data into training and test sets (30% held out for testing)
    val splitPer = args(5).toFloat
    val splits = data.randomSplit(Array(1-splitPer, splitPer))
    val (trainingData, testData) = (splits(0), splits(1))

    // Train a DecisionTree model.
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = args(2).toInt
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = args(3).toInt
    val maxBins = args(4).toInt

    val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)

    //println(model.toDebugString)
   // print(model)
    //print("\n")

    //model.save(sc , "/Users/sara/drp/myDTModelPath")


      model.save(sc , "file:" + args(1))
    } catch {
      case e: Exception => println("ERROR. tool unsuccessful:" + e);

    }
    // Evaluate model on test instances and compute test error
    //val labelAndPreds = testData.map { point =>
     // val prediction = model.predict(point.features)
     // (point.label, prediction)
    //}
    //val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    //println("Test Error = " + testErr)
    //println("Learned classification tree model:\n" + model.toDebugString)

  }

}
