package ml

import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import utils._

/**
 * Created by sara on 8/18/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object SVMsLearn {
  def main (args: Array[String]) {
    val filename = args(0)
    val output = args(1)
    val log = args(2)
    val numIterations = args(3).toInt

    FileLogger.open(log)
    try {
      val start = System.currentTimeMillis()
      //.split(";").map(s=>s.toInt)
      val conf = new SparkConf().setAppName("svmLearn")
      val sc = new SparkContext(conf)

      val metaFiles = FileHandler.loadInput(filename)
      val uritype = metaFiles(0).uriType
      val parsedData = sc.objectFile[LabeledPoint](metaFiles(0).uri).cache()
      // Load training data in LIBSVM format.
      //val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")

//////move to util/Data//////

//      val data = sc.textFile("data/mllib/ridge-data/lpsa.data")
//      val parsedData = data.map { line =>
//        val parts = line.split(',')
//        val len = parts.length
//        LabeledPoint(parts(len-1).toDouble, Vectors.dense(parts.slice(0,len-1).map(_.toDouble)))
//      }.cache()
///////////////////////////

      //val training = Data.loadData(sc, "file:/Users/sara/drp/data/kr_train.csv");

      //val objects = FileHandler.loadInput(filename);


      //val uritype = objects(0).uriType
      //val training = Data.loadData(sc, objects(0).uri)
      //val training = sc.objectFile[LabeledPoint](objects(0).uri)
      //training.saveAsObjectFile()

      //val test = Data.loadData(sc, "file:/Users/sara/drp/data/kr_test.csv");
      // Split data into training (60%) and test (40%).
      //val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
      //val training = splits(0).cache()
      //val test = splits(1)

      // Run training algorithm to build the mode
      //val numIterations = 100
      //training.map(l=>if (l.label < 0 ){ LabeledPoint(0, l.features) } else l )



      //paramMaps.addGrid(svm.numIterations, Array(10,20))

        //addGrid(svm.numIterations, Array(10,20))

      val model = SVMWithSGD.train(parsedData, numIterations)


      // Clear the default threshold.
      model.clearThreshold()



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

      model.save(sc, output)

      val stop = System.currentTimeMillis()
      // Compute raw scores on the test set.
      /*FileLogger.println("Threshold = " + (model.getThreshold match {case Some(x)=> x case None=> "None"}))
      val scoreAndLabels = training.map { point =>
        val score =if  (model.predict(point.features) >= 0.0) 1.0 else 0.0
        FileLogger.println(score + " " + point.label)
        (score, point.label)
      }

      val errors = scoreAndLabels.map(p=>if (p._1 == p._2 ) 0.0 else 1.0)

      val sum = errors.fold(0.0)((s,e)=> s + e)
      val accuracy = 1.0 - sum / errors.count()

      //print
      //scoreAndLabels.foreach( a=> println(a._1, a._2))
      //println("weights:")
      //model.weights.toArray.foreach(a=>println(a));


      // Get evaluation metrics.
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      val auROC = metrics.areaUnderROC()
      val pr = metrics.areaUnderPR()
      FileLogger.println("Accuracy = " + accuracy);
      FileLogger.println("Area under ROC = " + auROC)
      FileLogger.println("Area under pr = " + pr)
      // Save and load model
      model.save(sc, "file:" + args(1))
     */
     // val sameModel = SVMModel.load(sc, "myModelPath")

      FileLogger.println("SVM learn successfully done in " + ((stop - start) / 1000.0) + " sec" )
    }catch {
      case e: Exception => println(GalaxyException.getString(e));
    } finally {
      FileLogger.close();
    }
  }

}
