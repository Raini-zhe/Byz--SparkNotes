package ml

import java.io.{File, PrintWriter}

import org.apache.spark.mllib.classification.{ClassificationModel, NaiveBayesModel, LogisticRegressionModel, SVMModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.Saveable
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import utils.CCData

/**
 * Created by sara on 9/10/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object CCClassifier {


  def predict (sc: SparkContext, modelName :String, modelpath:String, data: RDD[LabeledPoint], yIndex : Int) : RDD[LabeledPoint] = {
    val model = if (modelName == "SVMWithSGD") {
      //println("numItr", numIterations)
      // println(training);
      SVMModel.load(sc, modelpath + "/" + yIndex.toString)
    } else if (modelName == "LogisticRegressionWithLBFGS") {
      LogisticRegressionModel.load(sc, modelpath + "/" + yIndex.toString)
    } else {
      // println(modelType);
      NaiveBayesModel.load(sc, modelpath + "/" + yIndex.toString)
    }
    val nData = data.map { point =>
      val score = if (model.predict(point.features) >= 0) 1.0 else 0.0
      LabeledPoint(-1, Vectors.dense(Array(score) ++ point.features.toArray))
    }
    if (yIndex > 0) {
      predict(sc, modelName, modelpath, nData, yIndex - 1)
    } else {
      data
    }
  }

  def main(args: Array[String]) {
    try {
      val conf = new SparkConf().setAppName("BRClassifier").setMaster("local")
      val sc = new SparkContext(conf)

      val filename = args(0)
      val modelName = args(4).toString
      val numClassAtr = args(5).toInt
      //val models = Array[ClassificationModel with Saveable](3);
      //val numClasses = args(5).toInt
      //val lambda = args(6).toFloat
      //val modelType = args(7)


      val labeled = args(3).toBoolean
      //val data = Data.loadData(sc, "file:/Users/sara/drp/data/kr_test.csv");
      val data = CCData.loadData(sc, numClassAtr, modelName, "file:" + filename);
      val modelpath = args(2)
      val pw = new PrintWriter(new File(args(1)));

      val finalData = predict(sc, modelName, modelpath, data(numClassAtr-1), numClassAtr-1)
      val predictedLabels = finalData.map(p=>p.features.toArray.slice(0,numClassAtr));
      predictedLabels.foreach(l=>{l.foreach(a=>{pw.print(a); pw.print(" ")}); pw.println("")} )
//      for (i <- 0 to NumClassAtr - 1 reverse) {
//
//        // val training = dataArray(i)
//        //println("model:",i);
//        //println(modelName);
//        val model = if (modelName == "SVMWithSGD") {
//          //println("numItr", numIterations)
//          // println(training);
//          SVMModel.load(sc, modelpath + "/" + i.toString)
//
//
//        } else if (modelName == "LogisticRegressionWithLBFGS") {
//
//          LogisticRegressionModel.load(sc, modelpath + "/" + i.toString)
//
//        } else {
//          // println(modelType);
//          NaiveBayesModel.load(sc, modelpath + "/" + i.toString)
//
//        }
//        //println("here3")
//        if (labeled) {
//          val scoreAndLabels = data(i).map { point =>
//            val score = if (model.predict(point.features) >= 0) 1.0 else 0.0
//            (score, point.label)
//          }
//          val ndata = data(i).map { point =>
//            val score = if (model.predict(point.features) >= 0) 1.0 else 0.0
//
//            (score, point.label)
//          }
//          scoreAndLabels.join()
//         // data(i-1).map(point => point.features.toArray.update(1,sc))
//          //val error = scoreAndLabels.map(a => if (a._1 == a._2) 1.0 else 0.0)
//          //val sum = error.fold(0.0)((s, p) => s + p)
//          //val accuracy = sum / error.count()
//          val accuracy = 1.0 * scoreAndLabels.filter(x => x._1 == x._2).count() / data(i).count()
//
//
//
//          for (a <- scoreAndLabels.collect()) {
//            pw.write(a._1 + " " + a._2 + "\n")
//          }
//          pw.write("accuracy = " + accuracy + "\n")
//
//
//        } else {
//          val scoreAndLabels = data(i).map { point =>
//            val score = model.predict(point.features)
//            (score)
//          }
//          //val pw = new PrintWriter(new File(args(1)));
//          for (a <- scoreAndLabels.collect()) {
//            pw.write(a + "\n")
//          }
//        }
//
//        //models.update(i, model)
//      }
//      //println("here2")
//      //for(x <- 0 to models.length - 1 ) {
//      //models(x).save(sc, "file:" + args(1) + "/" + x.toString);
      pw.close()



    }catch {
      case e: Exception => println("ERROR. tool unsuccessful:" + e);
    }
  }


}
