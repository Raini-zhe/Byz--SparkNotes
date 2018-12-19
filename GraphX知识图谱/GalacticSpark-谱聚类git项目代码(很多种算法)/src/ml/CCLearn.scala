package ml

/**
 * Created by sara on 9/10/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object CCLearn {

  import java.io.{File, PrintWriter}

  import org.apache.spark.mllib.classification._
  import org.apache.spark.mllib.linalg.Vectors
  import org.apache.spark.mllib.regression.LabeledPoint
  import org.apache.spark.mllib.util.Saveable
  import org.apache.spark.{SparkContext, SparkConf}
  import utils.CCData

  /**
   * Created by sara on 8/25/15.
   */
  object BRLearn {
    def main(args: Array[String]) {
      try {
        val conf = new SparkConf().setAppName("svmLearn").setMaster("local")
        val sc = new SparkContext(conf)

        val filename = args(0)
        val modelName = args(2).toString
        val NumClassAtr = args(3).toInt
        //val numIterations = 100
        val numIterations = args(4).toInt
        val numClasses = args(5).toInt
        val lambda = args(6).toFloat
        val modelType = args(7)


        val dataArray = CCData.loadData(sc, NumClassAtr, modelName, "file:" + filename)
        //val modelClass = if (modelName == "SVMWithSGD") { SVMModel.getClass } else if ( modelName == "LogisticRegressionWithLBFGS") {LogisticRegressionModel.getClass} else { NaiveBayesModel}
        //println("here1")

        val models = new Array[ClassificationModel with Saveable](NumClassAtr)
        //val models = if (modelName == "SVMWithSGD") { new Array[SVMModel](NumClassAtr) }
        //  else if ( modelName == "LogisticRegressionWithLBFGS") {
        //   new Array[LogisticRegressionModel](NumClassAtr)}
        //  else {  new Array[NaiveBayesModel](NumClassAtr)}
        for (i <- 0 to NumClassAtr-1) {

          val training = dataArray(i)
          //println("model:",i);
          //println(modelName);
          val model = if (modelName == "SVMWithSGD") {
            //println("numItr", numIterations)
            // println(training);
            SVMWithSGD.train(training, numIterations)
          } else if (modelName == "LogisticRegressionWithLBFGS") {
            new LogisticRegressionWithLBFGS()
              //.setNumClasses(10)
              .setNumClasses(numClasses)
              .run(training)
          } else {
            // println(modelType);
            NaiveBayes.train(training, lambda, modelType)
          }
          //println("here3")
          models.update(i, model)
        }
        //println("here2")
        for(x <- 0 to models.length - 1 ) {
          models(x).save(sc, "file:" + args(1) + "/" + x.toString);

        }

      }catch {
        case e: Exception => println("ERROR. tool unsuccessful:" + e);
      }
    }

  }


}
