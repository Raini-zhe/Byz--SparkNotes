/*
 * 规范化的XGB-Spark流程(可作编程参考模板)

 spark-submit --master yarn-cluster --num-executors 10 --executor-memory 6g --executor-cores 8 \
    --class me.codingcat.xgboost4j.AirlineClassifier --files conf/airline.conf \
     target/scala-2.11/xgboost4j-spark-scalability-assembly-0.1-SNAPSHOT.jar ./airline.conf
 */

package me.codingcat.xgboost4j

import java.io.File

import com.typesafe.config.{Config, ConfigFactory}
import me.codingcat.xgboost4j.common.Utils
import ml.dmlc.xgboost4j.scala.spark.{XGBoost, XGBoostEstimator, XGBoostModel}

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, SparkSession}

object AirlineClassifier {

  private def buildPreprocessingPipeline(): Pipeline = {
    // string indexers
    val monthIndexer = new StringIndexer().setInputCol("Month").setOutputCol("monthIdx")
    val daysOfMonthIndexer = new StringIndexer().setInputCol("DayOfMonth").
      setOutputCol("dayOfMonthIdx")
    val daysOfWeekIndexer = new StringIndexer().setInputCol("DayOfWeek").
      setOutputCol("daysOfWeekIdx")
    val uniqueCarrierIndexer = new StringIndexer().setInputCol("UniqueCarrier").setOutputCol(
      "uniqueCarrierIndex")
    val originIndexer = new StringIndexer().setInputCol("Origin").setOutputCol(
      "originIndexer")
    val destIndexer = new StringIndexer().setInputCol("Dest").setOutputCol(
      "destIndexer")
    // one-hot encoders
    val monthEncoder = new OneHotEncoder().setInputCol("monthIdx").
      setOutputCol("encodedMonth")
    val daysOfMonthEncoder = new OneHotEncoder().setInputCol("dayOfMonthIdx").
      setOutputCol("encodedDaysOfMonth")
    val daysOfWeekEncoder = new OneHotEncoder().setInputCol("daysOfWeekIdx").
      setOutputCol("encodedDaysOfWeek")
    val uniqueCarrierEncoder = new OneHotEncoder().setInputCol("uniqueCarrierIndex").
      setOutputCol("encodedCarrier")
    val originEncoder = new OneHotEncoder().setInputCol("originIndexer").
      setOutputCol("encodedOrigin")
    val destEncoder = new StringIndexer().setInputCol("destIndexer").setOutputCol(
      "encodedDest")


    val vectorAssembler = new VectorAssembler().setInputCols(
      Array("encodedMonth", "encodedDaysOfMonth", "encodedDaysOfWeek", "DepTime",
        "encodedCarrier", "encodedOrigin", "encodedDest", "Distance")
    ).setOutputCol("features")
    val pipeline = new Pipeline().setStages(
      Array(monthIndexer, daysOfMonthIndexer, daysOfWeekIndexer,
        uniqueCarrierIndexer, originIndexer, destIndexer, monthEncoder, daysOfMonthEncoder,
        daysOfWeekEncoder, uniqueCarrierEncoder, originEncoder, destEncoder, vectorAssembler))
    pipeline
  }

  private def runPreprocessingPipeline(pipeline: Pipeline, trainingSet: DataFrame): DataFrame = {
    pipeline.fit(trainingSet).transform(trainingSet).selectExpr(
      "features", "case when dep_delayed_15min = true then 1.0 else 0.0 end as label")
  }

  private def crossValidationWithXGBoost(
      xgbEstimator: XGBoostEstimator,
      trainingSet: DataFrame,
      tuningParamsPath: String): XGBoostModel = {
    val conf = ConfigFactory.parseFile(new File(tuningParamsPath))
    val paramGrid = new ParamGridBuilder()
      .addGrid(xgbEstimator.eta, Utils.fromConfigToParamGrid(conf)(xgbEstimator.eta.name))
      .addGrid(xgbEstimator.maxDepth, Utils.fromConfigToParamGrid(conf)(xgbEstimator.maxDepth.name).
        map(_.toInt))
      .addGrid(xgbEstimator.gamma, Utils.fromConfigToParamGrid(conf)(xgbEstimator.gamma.name))
      .addGrid(xgbEstimator.lambda, Utils.fromConfigToParamGrid(conf)(xgbEstimator.lambda.name))
      .addGrid(xgbEstimator.colSampleByTree, Utils.fromConfigToParamGrid(conf)(
        xgbEstimator.colSampleByTree.name))
      .addGrid(xgbEstimator.subSample, Utils.fromConfigToParamGrid(conf)(
        xgbEstimator.subSample.name))
      .build()
    val cv = new CrossValidator()
      .setEstimator(xgbEstimator)
      .setEvaluator(new BinaryClassificationEvaluator().
        setRawPredictionCol("probabilities").setLabelCol("label"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)
    val cvModel = cv.fit(trainingSet)
    cvModel.bestModel.asInstanceOf[XGBoostModel]
  }

  def main(args: Array[String]): Unit = {
    val config = ConfigFactory.parseFile(new File(args(0)))
    val libName = config.getString("me.codingcat.xgboost4j.lib")
    val trainingPath = config.getString("me.codingcat.xgboost4j.airline.trainingPath")
    val trainingRounds = config.getInt("me.codingcat.xgboost4j.rounds")
    val numWorkers = config.getInt("me.codingcat.xgboost4j.numWorkers")
    val treeType = config.getString("me.codingcat.xgboost4j.treeMethod")
    val sampleRate = config.getDouble("me.codingcat.xgboost4j.sampleRate")
    val params = Utils.fromConfigToXGBParams(config)
    val spark = SparkSession.builder().getOrCreate()
    val completeSet = spark.read.parquet(trainingPath)
    val sampledDataset = if (sampleRate > 0) {
      completeSet.sample(withReplacement = false, sampleRate)
    } else {
      completeSet
    }
    val Array(trainingSet, testSet) = sampledDataset.randomSplit(Array(0.8, 0.2))

    val pipeline = buildPreprocessingPipeline()
    val transformedTrainingSet = runPreprocessingPipeline(pipeline, trainingSet)
    val transformedTestset = runPreprocessingPipeline(pipeline, testSet)

    if (libName == "xgboost") {
      if (args.length >= 2) {
        val xgbEstimator = new XGBoostEstimator(params)
        xgbEstimator.set(xgbEstimator.round, trainingRounds)
        xgbEstimator.set(xgbEstimator.nWorkers, numWorkers)
        xgbEstimator.set(xgbEstimator.treeMethod, treeType)
        val bestModel = crossValidationWithXGBoost(xgbEstimator, transformedTrainingSet, args(1))
        println(s"best model: ${bestModel.extractParamMap()}")
        val eval = new BinaryClassificationEvaluator().setRawPredictionCol("prediction")
        println("eval results: " + eval.evaluate(bestModel.transform(transformedTestset)))
      } else {
        // directly training
        transformedTrainingSet.cache().foreach(_ => Unit)
        val startTime = System.nanoTime()
        val xgbModel = XGBoost.trainWithDataFrame(transformedTrainingSet, round = trainingRounds,
          nWorkers = numWorkers, params = Utils.fromConfigToXGBParams(config))
        println(s"===training time cost: ${(System.nanoTime() - startTime) / 1000.0 / 1000.0} ms")
        val resultDF = xgbModel.transform(transformedTestset)
        val binaryClassificationEvaluator = new BinaryClassificationEvaluator()
        binaryClassificationEvaluator.setRawPredictionCol("probabilities").setLabelCol("label")
        println(s"=====test AUC: ${binaryClassificationEvaluator.evaluate(resultDF)}======")
      }
    } else {
      val gradientBoostedTrees = new GBTClassifier()
      gradientBoostedTrees.setMaxBins(1000)
      gradientBoostedTrees.setMaxIter(500)
      gradientBoostedTrees.setMaxDepth(6)
      gradientBoostedTrees.setStepSize(1.0)
      transformedTrainingSet.cache().foreach(_ => Unit)
      val startTime = System.nanoTime()
      val model = gradientBoostedTrees.fit(transformedTrainingSet)
      println(s"===training time cost: ${(System.nanoTime() - startTime) / 1000.0 / 1000.0} ms")
      val resultDF = model.transform(transformedTestset)
      val binaryClassificationEvaluator = new BinaryClassificationEvaluator()
      binaryClassificationEvaluator.setRawPredictionCol("prediction").setLabelCol("label")
      println(s"=====test AUC: ${binaryClassificationEvaluator.evaluate(resultDF)}======")
    }
  }
}


/** me/codingcat/xgboost4j/common/Utils.scala ---------------------------------------------------------------------------------- */



/*
 * 参数文件
 */

package me.codingcat.xgboost4j.common

import scala.collection.mutable

import com.typesafe.config.Config

private[xgboost4j] object Utils { // private[xgboost4j]: 只能在这个包下使用 

  private val params = Map(
    "max_depth" -> 6,
    "min_child_weight" -> 1,
    "gamma" -> 0,
    "subsample" -> 1,
    "colsample_bytree" -> 1,
    "lambda" -> 1,
    "scale_pos_weight" -> 1,
    "silent" -> 0,
    "eta" -> 0.3,
    "objective" -> "binary:logistic"
  )

  private val paramSets = Map(
    "max_depth" -> "6",
    "min_child_weight" -> "1",
    "gamma" -> "0",
    "subsample" -> "1",
    "colsample_bytree" -> "1",
    "scale_pos_weight" -> "1",
    "silent" -> "0",
    "eta" -> "0.3",
    "lambda" -> "1"
  )

  def fromConfigToParamGrid(config: Config): Map[String, Array[Double]] = {
    val specifiedMap = new mutable.HashMap[String, Array[Double]]
    for (name <- paramSets.keys) {
      if (config.hasPath(s"me.codingcat.xgboost4j.$name")) {
        specifiedMap += name ->
          config.getString(s"me.codingcat.xgboost4j.$name").split(",").map(_.toDouble)
      }
    }
    specifiedMap.toMap
  }

  def fromConfigToXGBParams(config: Config): Map[String, Any] = {
    val specifiedMap = new mutable.HashMap[String, Any]
    for (name <- params.keys) {
      if (config.hasPath(s"me.codingcat.xgboost4j.$name")) {
        specifiedMap += name -> config.getAnyRef(s"me.codingcat.xgboost4j.$name").asInstanceOf[Any]  // 直接把paramSets 用 List toMap 不就行了?!
      }
    }
    params ++ specifiedMap.toMap
  }
}


































