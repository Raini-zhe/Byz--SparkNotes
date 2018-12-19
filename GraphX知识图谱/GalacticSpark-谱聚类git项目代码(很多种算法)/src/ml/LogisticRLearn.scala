package ml

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{TrainValidationSplit, CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf}
import org.apache.spark.SparkContext
import utils.{GalaxyException, FileHandler, Data, FileLogger}
import org.apache.spark.ml.classification.{LogisticRegression}

/**
 * Created by sara on 8/20/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object LogisticRLearn {
  def main(args: Array[String]) {
    val filename = args(0)
    val output = args(1);
    val log = args(2);

    FileLogger.open(log)

    val numClasses = args(3).toInt
    val validationType = args(4)
    val foldsStr = args(5)
    val regParamStr = args(6)
    val elasticParamStr = args(7)
    val iterNumStr = args(8)

    val filename2 = args(9)

    FileLogger.println("filename2 " + filename2)
    try {

      val conf = new SparkConf().setAppName("lrLearn")
      val sc = new SparkContext(conf)
      val sqlC = new SQLContext(sc)
      // Load training data in LIBSVM format.
      //val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")

      //val training = Data.loadData(sc, "data/mllib/sample_libsvm_data.txt");

      val objects = FileHandler.loadInput(filename);
      val uriType = objects(0).uriType
      val trainStr = Data.loadCSV(sqlC, objects(0).uri, true, false)
      //val training = sc.objectFile[LabeledPoint](objects(0).uri).cache()



      val trainingRDD = trainStr.map{ r => val s = r.getString(r.fieldIndex("features"));
           val sub = s.substring(1, s.length-1) //delete brackets

             val dar = sub.split(",").map(_.toDouble)
            val label = r.getString(r.fieldIndex("label")).toDouble
        (label, Vectors.dense(dar))
      }
      FileLogger.println("Feature Size:" + trainingRDD.take(10).apply(2)._2.size)
      //val trainingRDD = trainStr.map{r => val l = r.getList[Double](r.fieldIndex("features"));  (r.getAs[Double]("label"), Vectors.dense(.toArray())))
      val training = sqlC.createDataFrame(trainingRDD).toDF("label", "features").cache()
      training.count()
      //FileLogger.println(trainingRDD.first()._1 + " " + trainingRDD.first()._2.toArray.toString)


      val object2 = FileHandler.loadInput(filename2);
      val testStr = Data.loadCSV(sqlC, object2(0).uri, true, false)

      //val testRDD = testStr.map(r => (r.getAs[Double]("label"), Vectors.dense(r.getAs[("features").split(",").map(_.toDouble))))
      val testRDD = testStr.map{ r => val s = r.getString(r.fieldIndex("features"));
        val dar = s.substring(1, s.length-1).split(",").map(_.toDouble) //delete brackets
        val label = r.getString(r.fieldIndex("label")).toDouble
        (label, Vectors.dense(dar))
        //(r.getAs[Double]("label"), Vectors.dense(dar))
      }
      val test = sqlC.createDataFrame(testRDD).toDF("label", "features")

      val regParamAr = regParamStr.split(",").map(_.stripMargin).map(_.toDouble)
      val elasticParamAr = elasticParamStr.split(",").map(_.stripMargin).map(_.toDouble)
      val iterNumAr = iterNumStr.split(",").map(_.stripMargin).map(_.toInt)

      val lr = new LogisticRegression()
      //val pipeline = new Pipeline().setStages(Array(lr))


      //val m = pipeline.fit(training)
      val paramGrid = new ParamGridBuilder()
        .addGrid(lr.regParam, regParamAr)
        .addGrid(lr.elasticNetParam, elasticParamAr)
        .addGrid(lr.maxIter, iterNumAr)
        .build()

      val m = if (validationType == "cv") {
        val foldNum = foldsStr.toInt
        FileLogger.println("Running Cross-Validation with " + foldNum + " folds");
        val cv = new CrossValidator()
          .setEstimator(lr)
          .setEvaluator(new BinaryClassificationEvaluator)
          .setEstimatorParamMaps(paramGrid)
          .setNumFolds(foldNum)
        val cvModel = cv.fit(training)
        cvModel.write.save(output)
        cvModel
      } else if (validationType == "tv") {
        FileLogger.println("Running using train-validation split with " + foldsStr.toDouble + " ratio");
        val trainValidationSplit = new TrainValidationSplit()
          .setEstimator(lr)
          .setEvaluator(new BinaryClassificationEvaluator())
          .setEstimatorParamMaps(paramGrid)
          .setTrainRatio(foldsStr.toDouble)
        val model = trainValidationSplit.fit(training)
        model
        // model.save(output)
      } else {
        FileLogger.println("Running witout validation part using first assigned parameters.")
        FileLogger.println("RegParam:" + regParamAr.apply(0))

        lr.setElasticNetParam(elasticParamAr.apply(0))
        lr.setRegParam(regParamAr.apply(0))
        lr.setMaxIter(iterNumAr.apply(0))
        FileLogger.println(training.collect().mkString("\n"));
        val model = lr.fit(training)
        model.write.save(output);
        model
      }
      val results = m.transform(test)
      FileLogger.println(results.columns.mkString(","))


      val evaluator = new BinaryClassificationEvaluator()
        .setLabelCol("label")
        .setRawPredictionCol("rawPrediction")
       //.setMetricName("precision")
      //FileLogger.println(results.select("rawPrediction").collect().mkString("\n"));

      results.cache()
      val tp = results.select("label","prediction").map(r=>if (r.getDouble(0) == r.getDouble(1)) 1.0 else 0.0 ).reduce(_+_)

      val total= results.count();

      val naccuracy = tp / total
      FileLogger.println("nAccuracy = " + naccuracy);

      //FileLogger.println("tp = " + tp )
      val accuracy = evaluator.evaluate(results)
      FileLogger.println("result columns:" + results.columns.mkString(","))


      FileLogger.println("Test data accuaracy:" + accuracy);
      // Split data into training (60%) and test (40%).
      //val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
      //val training = splits(0).cache()
      //val test = splits(1)

      //val paramMaps = new ParamGridBuilder()
      // paramMaps.addGrid(lr.reg, Array(10,20))
      // Run training algorithm to build the model


      //LogisticRegressionWithGSD
      //val model = LogisticRegressionW setNumClasses(10)
      // .setNumClasses(numClasses)
      //.run(training)


      // Compute raw scores on the test set.
      //      val predictionAndLabels = training.map { case LabeledPoint(label, features) =>
      //        val prediction = model.predict(features)
      //        (prediction, label)
      //      }
      //
      //      // Get evaluation metrics.
      //      val metrics = new MulticlassMetrics(predictionAndLabels)
      //      val precision = metrics.precision
      //      println("Precision = " + precision)


      // Save and load model
      //model.save(sc, "myModelPath")

      //val sameModel = LogisticRegressionModel.load(sc, "myModelPath")

    }catch {
      case e: Exception => FileLogger.println(GalaxyException.getString(e))
    } finally {
      FileLogger.close()
    }
  }

}
