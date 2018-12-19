import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Column
import org.apache.spark.sql.DataFrameReader
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.DataFrameStatFunctions
import org.apache.spark.sql.functions._

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// ================（导入数据源）================ //

val spark = SparkSession.builder().appName("Spark Multilayer perceptron classifier").config("spark.some.config.option", "some-value").getOrCreate()

// For implicit conversions like converting RDDs to DataFrames
import spark.implicits._

val dataList: List[(Double, String, Double, Double, String, Double, Double, Double, Double)] = List(
      (0, "male", 37, 10, "no", 3, 18, 7, 4),
      (0, "female", 27, 4, "no", 4, 14, 6, 4),
      (0, "female", 32, 15, "yes", 1, 12, 1, 4),
      (0, "male", 57, 15, "yes", 5, 18, 6, 5),
      (0, "male", 22, 0.75, "no", 2, 17, 6, 3),
      (0, "female", 32, 1.5, "no", 2, 17, 5, 5),
      (0, "female", 22, 0.75, "no", 2, 12, 1, 3),
      (0, "male", 57, 15, "yes", 2, 14, 4, 4),
      (0, "female", 32, 15, "yes", 4, 16, 1, 2),
      (0, "male", 22, 1.5, "no", 4, 14, 4, 5),
      (0, "male", 37, 15, "yes", 2, 20, 7, 2),
      (0, "male", 27, 4, "yes", 4, 18, 6, 4),
      (0, "male", 47, 15, "yes", 5, 17, 6, 4),
      (0, "female", 22, 1.5, "no", 2, 17, 5, 4),
      (0, "female", 27, 4, "no", 4, 14, 5, 4),
      (0, "female", 37, 15, "yes", 1, 17, 5, 5),
      (0, "female", 37, 15, "yes", 2, 18, 4, 3),
      (0, "female", 22, 0.75, "no", 3, 16, 5, 4),
      (0, "female", 22, 1.5, "no", 2, 16, 5, 5),
      (0, "female", 27, 10, "yes", 2, 14, 1, 5),
      (0, "female", 22, 1.5, "no", 2, 16, 5, 5),
      (0, "female", 22, 1.5, "no", 2, 16, 5, 5),
      (0, "female", 27, 10, "yes", 4, 16, 5, 4),
      (0, "female", 32, 10, "yes", 3, 14, 1, 5),
      (0, "male", 37, 4, "yes", 2, 20, 6, 4),
      (0, "female", 22, 1.5, "no", 2, 18, 5, 5),
      (0, "female", 27, 7, "no", 4, 16, 1, 5),
      (0, "male", 42, 15, "yes", 5, 20, 6, 4),
      (0, "male", 27, 4, "yes", 3, 16, 5, 5),
      (0, "female", 27, 4, "yes", 3, 17, 5, 4),
      (0, "male", 42, 15, "yes", 4, 20, 6, 3),
      (0, "female", 22, 1.5, "no", 3, 16, 5, 5),
      (0, "male", 27, 0.417, "no", 4, 17, 6, 4),
      (0, "female", 42, 15, "yes", 5, 14, 5, 4),
      (0, "male", 32, 4, "yes", 1, 18, 6, 4),
      (0, "female", 22, 1.5, "no", 4, 16, 5, 3),
      (0, "female", 42, 15, "yes", 3, 12, 1, 4),
      (0, "female", 22, 4, "no", 4, 17, 5, 5),
      (0, "male", 22, 1.5, "yes", 1, 14, 3, 5),
      (0, "female", 22, 0.75, "no", 3, 16, 1, 5),
      (0, "male", 32, 10, "yes", 5, 20, 6, 5),
      (0, "male", 52, 15, "yes", 5, 18, 6, 3),
      (0, "female", 22, 0.417, "no", 5, 14, 1, 4),
      (0, "female", 27, 4, "yes", 2, 18, 6, 1),
      (0, "female", 32, 7, "yes", 5, 17, 5, 3),
      (0, "male", 22, 4, "no", 3, 16, 5, 5),
      (0, "female", 27, 7, "yes", 4, 18, 6, 5),
      (0, "female", 42, 15, "yes", 2, 18, 5, 4),
      (0, "male", 27, 1.5, "yes", 4, 16, 3, 5),
      (0, "male", 42, 15, "yes", 2, 20, 6, 4),
      (0, "female", 22, 0.75, "no", 5, 14, 3, 5),
      (0, "male", 32, 7, "yes", 2, 20, 6, 4),
      (0, "male", 27, 4, "yes", 5, 20, 6, 5),
      (0, "male", 27, 10, "yes", 4, 20, 6, 4),
      (0, "male", 22, 4, "no", 1, 18, 5, 5),
      (0, "female", 37, 15, "yes", 4, 14, 3, 1),
      (0, "male", 22, 1.5, "yes", 5, 16, 4, 4),
      (0, "female", 37, 15, "yes", 4, 17, 1, 5),
      (0, "female", 27, 0.75, "no", 4, 17, 5, 4),
      (0, "male", 32, 10, "yes", 4, 20, 6, 4),
      (0, "female", 47, 15, "yes", 5, 14, 7, 2),
      (0, "male", 37, 10, "yes", 3, 20, 6, 4),
      (0, "female", 22, 0.75, "no", 2, 16, 5, 5),
      (0, "male", 27, 4, "no", 2, 18, 4, 5),
      (0, "male", 32, 7, "no", 4, 20, 6, 4),
      (0, "male", 42, 15, "yes", 2, 17, 3, 5),
      (0, "male", 37, 10, "yes", 4, 20, 6, 4),
      (0, "female", 47, 15, "yes", 3, 17, 6, 5),
      (0, "female", 22, 1.5, "no", 5, 16, 5, 5),
      (0, "female", 27, 1.5, "no", 2, 16, 6, 4),
      (0, "female", 27, 4, "no", 3, 17, 5, 5),
      (0, "female", 32, 10, "yes", 5, 14, 4, 5),
      (0, "female", 22, 0.125, "no", 2, 12, 5, 5),
      (0, "male", 47, 15, "yes", 4, 14, 4, 3),
      (0, "male", 32, 15, "yes", 1, 14, 5, 5),
      (0, "male", 27, 7, "yes", 4, 16, 5, 5),
      (0, "female", 22, 1.5, "yes", 3, 16, 5, 5),
      (0, "male", 27, 4, "yes", 3, 17, 6, 5),
      (0, "female", 22, 1.5, "no", 3, 16, 5, 5),
      (0, "male", 57, 15, "yes", 2, 14, 7, 2),
      (0, "male", 17.5, 1.5, "yes", 3, 18, 6, 5),
      (0, "male", 57, 15, "yes", 4, 20, 6, 5),
      (0, "female", 22, 0.75, "no", 2, 16, 3, 4),
      (0, "male", 42, 4, "no", 4, 17, 3, 3),
      (0, "female", 22, 1.5, "yes", 4, 12, 1, 5),
      (0, "female", 22, 0.417, "no", 1, 17, 6, 4),
      (0, "female", 32, 15, "yes", 4, 17, 5, 5),
      (0, "female", 27, 1.5, "no", 3, 18, 5, 2),
      (0, "female", 22, 1.5, "yes", 3, 14, 1, 5),
      (0, "female", 37, 15, "yes", 3, 14, 1, 4),
      (0, "female", 32, 15, "yes", 4, 14, 3, 4),
      (0, "male", 37, 10, "yes", 2, 14, 5, 3),
      (0, "male", 37, 10, "yes", 4, 16, 5, 4),
      (0, "male", 57, 15, "yes", 5, 20, 5, 3),
      (0, "male", 27, 0.417, "no", 1, 16, 3, 4),
      (0, "female", 42, 15, "yes", 5, 14, 1, 5),
      (0, "male", 57, 15, "yes", 3, 16, 6, 1),
      (0, "male", 37, 10, "yes", 1, 16, 6, 4),
      (0, "male", 37, 15, "yes", 3, 17, 5, 5),
      (0, "male", 37, 15, "yes", 4, 20, 6, 5),
      (0, "female", 27, 10, "yes", 5, 14, 1, 5),
      (0, "male", 37, 10, "yes", 2, 18, 6, 4),
      (0, "female", 22, 0.125, "no", 4, 12, 4, 5),
      (0, "male", 57, 15, "yes", 5, 20, 6, 5),
      (0, "female", 37, 15, "yes", 4, 18, 6, 4),
      (0, "male", 22, 4, "yes", 4, 14, 6, 4),
      (0, "male", 27, 7, "yes", 4, 18, 5, 4),
      (0, "male", 57, 15, "yes", 4, 20, 5, 4),
      (0, "male", 32, 15, "yes", 3, 14, 6, 3),
      (0, "female", 22, 1.5, "no", 2, 14, 5, 4),
      (0, "female", 32, 7, "yes", 4, 17, 1, 5),
      (0, "female", 37, 15, "yes", 4, 17, 6, 5),
      (0, "female", 32, 1.5, "no", 5, 18, 5, 5),
      (0, "male", 42, 10, "yes", 5, 20, 7, 4),
      (0, "female", 27, 7, "no", 3, 16, 5, 4),
      (0, "male", 37, 15, "no", 4, 20, 6, 5),
      (0, "male", 37, 15, "yes", 4, 14, 3, 2),
      (0, "male", 32, 10, "no", 5, 18, 6, 4),
      (0, "female", 22, 0.75, "no", 4, 16, 1, 5),
      (0, "female", 27, 7, "yes", 4, 12, 2, 4),
      (0, "female", 27, 7, "yes", 2, 16, 2, 5),
      (0, "female", 42, 15, "yes", 5, 18, 5, 4),
      (0, "male", 42, 15, "yes", 4, 17, 5, 3),
      (0, "female", 27, 7, "yes", 2, 16, 1, 2),
      (0, "female", 22, 1.5, "no", 3, 16, 5, 5),
      (0, "male", 37, 15, "yes", 5, 20, 6, 5),
      (0, "female", 22, 0.125, "no", 2, 14, 4, 5),
      (0, "male", 27, 1.5, "no", 4, 16, 5, 5),
      (0, "male", 32, 1.5, "no", 2, 18, 6, 5),
      (0, "male", 27, 1.5, "no", 2, 17, 6, 5),
      (0, "female", 27, 10, "yes", 4, 16, 1, 3),
      (0, "male", 42, 15, "yes", 4, 18, 6, 5),
      (0, "female", 27, 1.5, "no", 2, 16, 6, 5),
      (0, "male", 27, 4, "no", 2, 18, 6, 3),
      (0, "female", 32, 10, "yes", 3, 14, 5, 3),
      (0, "female", 32, 15, "yes", 3, 18, 5, 4),
      (0, "female", 22, 0.75, "no", 2, 18, 6, 5),
      (0, "female", 37, 15, "yes", 2, 16, 1, 4),
      (0, "male", 27, 4, "yes", 4, 20, 5, 5),
      (0, "male", 27, 4, "no", 1, 20, 5, 4),
      (0, "female", 27, 10, "yes", 2, 12, 1, 4),
      (0, "female", 32, 15, "yes", 5, 18, 6, 4),
      (0, "male", 27, 7, "yes", 5, 12, 5, 3),
      (0, "male", 52, 15, "yes", 2, 18, 5, 4),
      (0, "male", 27, 4, "no", 3, 20, 6, 3),
      (0, "male", 37, 4, "yes", 1, 18, 5, 4),
      (0, "male", 27, 4, "yes", 4, 14, 5, 4),
      (0, "female", 52, 15, "yes", 5, 12, 1, 3),
      (0, "female", 57, 15, "yes", 4, 16, 6, 4),
      (0, "male", 27, 7, "yes", 1, 16, 5, 4),
      (0, "male", 37, 7, "yes", 4, 20, 6, 3),
      (0, "male", 22, 0.75, "no", 2, 14, 4, 3),
      (0, "male", 32, 4, "yes", 2, 18, 5, 3),
      (0, "male", 37, 15, "yes", 4, 20, 6, 3),
      (0, "male", 22, 0.75, "yes", 2, 14, 4, 3),
      (0, "male", 42, 15, "yes", 4, 20, 6, 3),
      (0, "female", 52, 15, "yes", 5, 17, 1, 1),
      (0, "female", 37, 15, "yes", 4, 14, 1, 2),
      (0, "male", 27, 7, "yes", 4, 14, 5, 3),
      (0, "male", 32, 4, "yes", 2, 16, 5, 5),
      (0, "female", 27, 4, "yes", 2, 18, 6, 5),
      (0, "female", 27, 4, "yes", 2, 18, 5, 5),
      (0, "male", 37, 15, "yes", 5, 18, 6, 5),
      (0, "female", 47, 15, "yes", 5, 12, 5, 4),
      (0, "female", 32, 10, "yes", 3, 17, 1, 4),
      (0, "female", 27, 1.5, "yes", 4, 17, 1, 2),
      (0, "female", 57, 15, "yes", 2, 18, 5, 2),
      (0, "female", 22, 1.5, "no", 4, 14, 5, 4),
      (0, "male", 42, 15, "yes", 3, 14, 3, 4),
      (0, "male", 57, 15, "yes", 4, 9, 2, 2),
      (0, "male", 57, 15, "yes", 4, 20, 6, 5),
      (0, "female", 22, 0.125, "no", 4, 14, 4, 5),
      (0, "female", 32, 10, "yes", 4, 14, 1, 5),
      (0, "female", 42, 15, "yes", 3, 18, 5, 4),
      (0, "female", 27, 1.5, "no", 2, 18, 6, 5),
      (0, "male", 32, 0.125, "yes", 2, 18, 5, 2),
      (0, "female", 27, 4, "no", 3, 16, 5, 4),
      (0, "female", 27, 10, "yes", 2, 16, 1, 4),
      (0, "female", 32, 7, "yes", 4, 16, 1, 3),
      (0, "female", 37, 15, "yes", 4, 14, 5, 4),
      (0, "female", 42, 15, "yes", 5, 17, 6, 2),
      (0, "male", 32, 1.5, "yes", 4, 14, 6, 5),
      (0, "female", 32, 4, "yes", 3, 17, 5, 3),
      (0, "female", 37, 7, "no", 4, 18, 5, 5),
      (0, "female", 22, 0.417, "yes", 3, 14, 3, 5),
      (0, "female", 27, 7, "yes", 4, 14, 1, 5),
      (0, "male", 27, 0.75, "no", 3, 16, 5, 5),
      (0, "male", 27, 4, "yes", 2, 20, 5, 5),
      (0, "male", 32, 10, "yes", 4, 16, 4, 5),
      (0, "male", 32, 15, "yes", 1, 14, 5, 5),
      (0, "male", 22, 0.75, "no", 3, 17, 4, 5),
      (0, "female", 27, 7, "yes", 4, 17, 1, 4),
      (0, "male", 27, 0.417, "yes", 4, 20, 5, 4),
      (0, "male", 37, 15, "yes", 4, 20, 5, 4),
      (0, "female", 37, 15, "yes", 2, 14, 1, 3),
      (0, "male", 22, 4, "yes", 1, 18, 5, 4),
      (0, "male", 37, 15, "yes", 4, 17, 5, 3),
      (0, "female", 22, 1.5, "no", 2, 14, 4, 5),
      (0, "male", 52, 15, "yes", 4, 14, 6, 2),
      (0, "female", 22, 1.5, "no", 4, 17, 5, 5),
      (0, "male", 32, 4, "yes", 5, 14, 3, 5),
      (0, "male", 32, 4, "yes", 2, 14, 3, 5),
      (0, "female", 22, 1.5, "no", 3, 16, 6, 5),
      (0, "male", 27, 0.75, "no", 2, 18, 3, 3),
      (0, "female", 22, 7, "yes", 2, 14, 5, 2),
      (0, "female", 27, 0.75, "no", 2, 17, 5, 3),
      (0, "female", 37, 15, "yes", 4, 12, 1, 2),
      (0, "female", 22, 1.5, "no", 1, 14, 1, 5),
      (0, "female", 37, 10, "no", 2, 12, 4, 4),
      (0, "female", 37, 15, "yes", 4, 18, 5, 3),
      (0, "female", 42, 15, "yes", 3, 12, 3, 3),
      (0, "male", 22, 4, "no", 2, 18, 5, 5),
      (0, "male", 52, 7, "yes", 2, 20, 6, 2),
      (0, "male", 27, 0.75, "no", 2, 17, 5, 5),
      (0, "female", 27, 4, "no", 2, 17, 4, 5),
      (0, "male", 42, 1.5, "no", 5, 20, 6, 5),
      (0, "male", 22, 1.5, "no", 4, 17, 6, 5),
      (0, "male", 22, 4, "no", 4, 17, 5, 3),
      (0, "female", 22, 4, "yes", 1, 14, 5, 4),
      (0, "male", 37, 15, "yes", 5, 20, 4, 5),
      (0, "female", 37, 10, "yes", 3, 16, 6, 3),
      (0, "male", 42, 15, "yes", 4, 17, 6, 5),
      (0, "female", 47, 15, "yes", 4, 17, 5, 5),
      (0, "male", 22, 1.5, "no", 4, 16, 5, 4),
      (0, "female", 32, 10, "yes", 3, 12, 1, 4),
      (0, "female", 22, 7, "yes", 1, 14, 3, 5),
      (0, "female", 32, 10, "yes", 4, 17, 5, 4),
      (0, "male", 27, 1.5, "yes", 2, 16, 2, 4),
      (0, "male", 37, 15, "yes", 4, 14, 5, 5),
      (0, "male", 42, 4, "yes", 3, 14, 4, 5),
      (0, "female", 37, 15, "yes", 5, 14, 5, 4),
      (0, "female", 32, 7, "yes", 4, 17, 5, 5),
      (0, "female", 42, 15, "yes", 4, 18, 6, 5),
      (0, "male", 27, 4, "no", 4, 18, 6, 4),
      (0, "male", 22, 0.75, "no", 4, 18, 6, 5),
      (0, "male", 27, 4, "yes", 4, 14, 5, 3),
      (0, "female", 22, 0.75, "no", 5, 18, 1, 5),
      (0, "female", 52, 15, "yes", 5, 9, 5, 5),
      (0, "male", 32, 10, "yes", 3, 14, 5, 5),
      (0, "female", 37, 15, "yes", 4, 16, 4, 4),
      (0, "male", 32, 7, "yes", 2, 20, 5, 4),
      (0, "female", 42, 15, "yes", 3, 18, 1, 4),
      (0, "male", 32, 15, "yes", 1, 16, 5, 5),
      (0, "male", 27, 4, "yes", 3, 18, 5, 5),
      (0, "female", 32, 15, "yes", 4, 12, 3, 4),
      (0, "male", 22, 0.75, "yes", 3, 14, 2, 4),
      (0, "female", 22, 1.5, "no", 3, 16, 5, 3),
      (0, "female", 42, 15, "yes", 4, 14, 3, 5),
      (0, "female", 52, 15, "yes", 3, 16, 5, 4),
      (0, "male", 37, 15, "yes", 5, 20, 6, 4),
      (0, "female", 47, 15, "yes", 4, 12, 2, 3),
      (0, "male", 57, 15, "yes", 2, 20, 6, 4),
      (0, "male", 32, 7, "yes", 4, 17, 5, 5),
      (0, "female", 27, 7, "yes", 4, 17, 1, 4),
      (0, "male", 22, 1.5, "no", 1, 18, 6, 5),
      (0, "female", 22, 4, "yes", 3, 9, 1, 4),
      (0, "female", 22, 1.5, "no", 2, 14, 1, 5),
      (0, "male", 42, 15, "yes", 2, 20, 6, 4),
      (0, "male", 57, 15, "yes", 4, 9, 2, 4),
      (0, "female", 27, 7, "yes", 2, 18, 1, 5),
      (0, "female", 22, 4, "yes", 3, 14, 1, 5),
      (0, "male", 37, 15, "yes", 4, 14, 5, 3),
      (0, "male", 32, 7, "yes", 1, 18, 6, 4),
      (0, "female", 22, 1.5, "no", 2, 14, 5, 5),
      (0, "female", 22, 1.5, "yes", 3, 12, 1, 3),
      (0, "male", 52, 15, "yes", 2, 14, 5, 5),
      (0, "female", 37, 15, "yes", 2, 14, 1, 1),
      (0, "female", 32, 10, "yes", 2, 14, 5, 5),
      (0, "male", 42, 15, "yes", 4, 20, 4, 5),
      (0, "female", 27, 4, "yes", 3, 18, 4, 5),
      (0, "male", 37, 15, "yes", 4, 20, 6, 5),
      (0, "male", 27, 1.5, "no", 3, 18, 5, 5),
      (0, "female", 22, 0.125, "no", 2, 16, 6, 3),
      (0, "male", 32, 10, "yes", 2, 20, 6, 3),
      (0, "female", 27, 4, "no", 4, 18, 5, 4),
      (0, "female", 27, 7, "yes", 2, 12, 5, 1),
      (0, "male", 32, 4, "yes", 5, 18, 6, 3),
      (0, "female", 37, 15, "yes", 2, 17, 5, 5),
      (0, "male", 47, 15, "no", 4, 20, 6, 4),
      (0, "male", 27, 1.5, "no", 1, 18, 5, 5),
      (0, "male", 37, 15, "yes", 4, 20, 6, 4),
      (0, "female", 32, 15, "yes", 4, 18, 1, 4),
      (0, "female", 32, 7, "yes", 4, 17, 5, 4),
      (0, "female", 42, 15, "yes", 3, 14, 1, 3),
      (0, "female", 27, 7, "yes", 3, 16, 1, 4),
      (0, "male", 27, 1.5, "no", 3, 16, 4, 2),
      (0, "male", 22, 1.5, "no", 3, 16, 3, 5),
      (0, "male", 27, 4, "yes", 3, 16, 4, 2),
      (0, "female", 27, 7, "yes", 3, 12, 1, 2),
      (0, "female", 37, 15, "yes", 2, 18, 5, 4),
      (0, "female", 37, 7, "yes", 3, 14, 4, 4),
      (0, "male", 22, 1.5, "no", 2, 16, 5, 5),
      (0, "male", 37, 15, "yes", 5, 20, 5, 4),
      (0, "female", 22, 1.5, "no", 4, 16, 5, 3),
      (0, "female", 32, 10, "yes", 4, 16, 1, 5),
      (0, "male", 27, 4, "no", 2, 17, 5, 3),
      (0, "female", 22, 0.417, "no", 4, 14, 5, 5),
      (0, "female", 27, 4, "no", 2, 18, 5, 5),
      (0, "male", 37, 15, "yes", 4, 18, 5, 3),
      (0, "male", 37, 10, "yes", 5, 20, 7, 4),
      (0, "female", 27, 7, "yes", 2, 14, 4, 2),
      (0, "male", 32, 4, "yes", 2, 16, 5, 5),
      (0, "male", 32, 4, "yes", 2, 16, 6, 4),
      (0, "male", 22, 1.5, "no", 3, 18, 4, 5),
      (0, "female", 22, 4, "yes", 4, 14, 3, 4),
      (0, "female", 17.5, 0.75, "no", 2, 18, 5, 4),
      (0, "male", 32, 10, "yes", 4, 20, 4, 5),
      (0, "female", 32, 0.75, "no", 5, 14, 3, 3),
      (0, "male", 37, 15, "yes", 4, 17, 5, 3),
      (0, "male", 32, 4, "no", 3, 14, 4, 5),
      (0, "female", 27, 1.5, "no", 2, 17, 3, 2),
      (0, "female", 22, 7, "yes", 4, 14, 1, 5),
      (0, "male", 47, 15, "yes", 5, 14, 6, 5),
      (0, "male", 27, 4, "yes", 1, 16, 4, 4),
      (0, "female", 37, 15, "yes", 5, 14, 1, 3),
      (0, "male", 42, 4, "yes", 4, 18, 5, 5),
      (0, "female", 32, 4, "yes", 2, 14, 1, 5),
      (0, "male", 52, 15, "yes", 2, 14, 7, 4),
      (0, "female", 22, 1.5, "no", 2, 16, 1, 4),
      (0, "male", 52, 15, "yes", 4, 12, 2, 4),
      (0, "female", 22, 0.417, "no", 3, 17, 1, 5),
      (0, "female", 22, 1.5, "no", 2, 16, 5, 5),
      (0, "male", 27, 4, "yes", 4, 20, 6, 4),
      (0, "female", 32, 15, "yes", 4, 14, 1, 5),
      (0, "female", 27, 1.5, "no", 2, 16, 3, 5),
      (0, "male", 32, 4, "no", 1, 20, 6, 5),
      (0, "male", 37, 15, "yes", 3, 20, 6, 4),
      (0, "female", 32, 10, "no", 2, 16, 6, 5),
      (0, "female", 32, 10, "yes", 5, 14, 5, 5),
      (0, "male", 37, 1.5, "yes", 4, 18, 5, 3),
      (0, "male", 32, 1.5, "no", 2, 18, 4, 4),
      (0, "female", 32, 10, "yes", 4, 14, 1, 4),
      (0, "female", 47, 15, "yes", 4, 18, 5, 4),
      (0, "female", 27, 10, "yes", 5, 12, 1, 5),
      (0, "male", 27, 4, "yes", 3, 16, 4, 5),
      (0, "female", 37, 15, "yes", 4, 12, 4, 2),
      (0, "female", 27, 0.75, "no", 4, 16, 5, 5),
      (0, "female", 37, 15, "yes", 4, 16, 1, 5),
      (0, "female", 32, 15, "yes", 3, 16, 1, 5),
      (0, "female", 27, 10, "yes", 2, 16, 1, 5),
      (0, "male", 27, 7, "no", 2, 20, 6, 5),
      (0, "female", 37, 15, "yes", 2, 14, 1, 3),
      (0, "male", 27, 1.5, "yes", 2, 17, 4, 4),
      (0, "female", 22, 0.75, "yes", 2, 14, 1, 5),
      (0, "male", 22, 4, "yes", 4, 14, 2, 4),
      (0, "male", 42, 0.125, "no", 4, 17, 6, 4),
      (0, "male", 27, 1.5, "yes", 4, 18, 6, 5),
      (0, "male", 27, 7, "yes", 3, 16, 6, 3),
      (0, "female", 52, 15, "yes", 4, 14, 1, 3),
      (0, "male", 27, 1.5, "no", 5, 20, 5, 2),
      (0, "female", 27, 1.5, "no", 2, 16, 5, 5),
      (0, "female", 27, 1.5, "no", 3, 17, 5, 5),
      (0, "male", 22, 0.125, "no", 5, 16, 4, 4),
      (0, "female", 27, 4, "yes", 4, 16, 1, 5),
      (0, "female", 27, 4, "yes", 4, 12, 1, 5),
      (0, "female", 47, 15, "yes", 2, 14, 5, 5),
      (0, "female", 32, 15, "yes", 3, 14, 5, 3),
      (0, "male", 42, 7, "yes", 2, 16, 5, 5),
      (0, "male", 22, 0.75, "no", 4, 16, 6, 4),
      (0, "male", 27, 0.125, "no", 3, 20, 6, 5),
      (0, "male", 32, 10, "yes", 3, 20, 6, 5),
      (0, "female", 22, 0.417, "no", 5, 14, 4, 5),
      (0, "female", 47, 15, "yes", 5, 14, 1, 4),
      (0, "female", 32, 10, "yes", 3, 14, 1, 5),
      (0, "male", 57, 15, "yes", 4, 17, 5, 5),
      (0, "male", 27, 4, "yes", 3, 20, 6, 5),
      (0, "female", 32, 7, "yes", 4, 17, 1, 5),
      (0, "female", 37, 10, "yes", 4, 16, 1, 5),
      (0, "female", 32, 10, "yes", 1, 18, 1, 4),
      (0, "female", 22, 4, "no", 3, 14, 1, 4),
      (0, "female", 27, 7, "yes", 4, 14, 3, 2),
      (0, "male", 57, 15, "yes", 5, 18, 5, 2),
      (0, "male", 32, 7, "yes", 2, 18, 5, 5),
      (0, "female", 27, 1.5, "no", 4, 17, 1, 3),
      (0, "male", 22, 1.5, "no", 4, 14, 5, 5),
      (0, "female", 22, 1.5, "yes", 4, 14, 5, 4),
      (0, "female", 32, 7, "yes", 3, 16, 1, 5),
      (0, "female", 47, 15, "yes", 3, 16, 5, 4),
      (0, "female", 22, 0.75, "no", 3, 16, 1, 5),
      (0, "female", 22, 1.5, "yes", 2, 14, 5, 5),
      (0, "female", 27, 4, "yes", 1, 16, 5, 5),
      (0, "male", 52, 15, "yes", 4, 16, 5, 5),
      (0, "male", 32, 10, "yes", 4, 20, 6, 5),
      (0, "male", 47, 15, "yes", 4, 16, 6, 4),
      (0, "female", 27, 7, "yes", 2, 14, 1, 2),
      (0, "female", 22, 1.5, "no", 4, 14, 4, 5),
      (0, "female", 32, 10, "yes", 2, 16, 5, 4),
      (0, "female", 22, 0.75, "no", 2, 16, 5, 4),
      (0, "female", 22, 1.5, "no", 2, 16, 5, 5),
      (0, "female", 42, 15, "yes", 3, 18, 6, 4),
      (0, "female", 27, 7, "yes", 5, 14, 4, 5),
      (0, "male", 42, 15, "yes", 4, 16, 4, 4),
      (0, "female", 57, 15, "yes", 3, 18, 5, 2),
      (0, "male", 42, 15, "yes", 3, 18, 6, 2),
      (0, "female", 32, 7, "yes", 2, 14, 1, 2),
      (0, "male", 22, 4, "no", 5, 12, 4, 5),
      (0, "female", 22, 1.5, "no", 1, 16, 6, 5),
      (0, "female", 22, 0.75, "no", 1, 14, 4, 5),
      (0, "female", 32, 15, "yes", 4, 12, 1, 5),
      (0, "male", 22, 1.5, "no", 2, 18, 5, 3),
      (0, "male", 27, 4, "yes", 5, 17, 2, 5),
      (0, "female", 27, 4, "yes", 4, 12, 1, 5),
      (0, "male", 42, 15, "yes", 5, 18, 5, 4),
      (0, "male", 32, 1.5, "no", 2, 20, 7, 3),
      (0, "male", 57, 15, "no", 4, 9, 3, 1),
      (0, "male", 37, 7, "no", 4, 18, 5, 5),
      (0, "male", 52, 15, "yes", 2, 17, 5, 4),
      (0, "male", 47, 15, "yes", 4, 17, 6, 5),
      (0, "female", 27, 7, "no", 2, 17, 5, 4),
      (0, "female", 27, 7, "yes", 4, 14, 5, 5),
      (0, "female", 22, 4, "no", 2, 14, 3, 3),
      (0, "male", 37, 7, "yes", 2, 20, 6, 5),
      (0, "male", 27, 7, "no", 4, 12, 4, 3),
      (0, "male", 42, 10, "yes", 4, 18, 6, 4),
      (0, "female", 22, 1.5, "no", 3, 14, 1, 5),
      (0, "female", 22, 4, "yes", 2, 14, 1, 3),
      (0, "female", 57, 15, "no", 4, 20, 6, 5),
      (0, "male", 37, 15, "yes", 4, 14, 4, 3),
      (0, "female", 27, 7, "yes", 3, 18, 5, 5),
      (0, "female", 17.5, 10, "no", 4, 14, 4, 5),
      (0, "male", 22, 4, "yes", 4, 16, 5, 5),
      (0, "female", 27, 4, "yes", 2, 16, 1, 4),
      (0, "female", 37, 15, "yes", 2, 14, 5, 1),
      (0, "female", 22, 1.5, "no", 5, 14, 1, 4),
      (0, "male", 27, 7, "yes", 2, 20, 5, 4),
      (0, "male", 27, 4, "yes", 4, 14, 5, 5),
      (0, "male", 22, 0.125, "no", 1, 16, 3, 5),
      (0, "female", 27, 7, "yes", 4, 14, 1, 4),
      (0, "female", 32, 15, "yes", 5, 16, 5, 3),
      (0, "male", 32, 10, "yes", 4, 18, 5, 4),
      (0, "female", 32, 15, "yes", 2, 14, 3, 4),
      (0, "female", 22, 1.5, "no", 3, 17, 5, 5),
      (0, "male", 27, 4, "yes", 4, 17, 4, 4),
      (0, "female", 52, 15, "yes", 5, 14, 1, 5),
      (0, "female", 27, 7, "yes", 2, 12, 1, 2),
      (0, "female", 27, 7, "yes", 3, 12, 1, 4),
      (0, "female", 42, 15, "yes", 2, 14, 1, 4),
      (0, "female", 42, 15, "yes", 4, 14, 5, 4),
      (0, "male", 27, 7, "yes", 4, 14, 3, 3),
      (0, "male", 27, 7, "yes", 2, 20, 6, 2),
      (0, "female", 42, 15, "yes", 3, 12, 3, 3),
      (0, "male", 27, 4, "yes", 3, 16, 3, 5),
      (0, "female", 27, 7, "yes", 3, 14, 1, 4),
      (0, "female", 22, 1.5, "no", 2, 14, 4, 5),
      (0, "female", 27, 4, "yes", 4, 14, 1, 4),
      (0, "female", 22, 4, "no", 4, 14, 5, 5),
      (0, "female", 22, 1.5, "no", 2, 16, 4, 5),
      (0, "male", 47, 15, "no", 4, 14, 5, 4),
      (0, "male", 37, 10, "yes", 2, 18, 6, 2),
      (0, "male", 37, 15, "yes", 3, 17, 5, 4),
      (0, "female", 27, 4, "yes", 2, 16, 1, 4),
      (3, "male", 27, 1.5, "no", 3, 18, 4, 4),
      (3, "female", 27, 4, "yes", 3, 17, 1, 5),
      (7, "male", 37, 15, "yes", 5, 18, 6, 2),
      (12, "female", 32, 10, "yes", 3, 17, 5, 2),
      (1, "male", 22, 0.125, "no", 4, 16, 5, 5),
      (1, "female", 22, 1.5, "yes", 2, 14, 1, 5),
      (12, "male", 37, 15, "yes", 4, 14, 5, 2),
      (7, "female", 22, 1.5, "no", 2, 14, 3, 4),
      (2, "male", 37, 15, "yes", 2, 18, 6, 4),
      (3, "female", 32, 15, "yes", 4, 12, 3, 2),
      (1, "female", 37, 15, "yes", 4, 14, 4, 2),
      (7, "female", 42, 15, "yes", 3, 17, 1, 4),
      (12, "female", 42, 15, "yes", 5, 9, 4, 1),
      (12, "male", 37, 10, "yes", 2, 20, 6, 2),
      (12, "female", 32, 15, "yes", 3, 14, 1, 2),
      (3, "male", 27, 4, "no", 1, 18, 6, 5),
      (7, "male", 37, 10, "yes", 2, 18, 7, 3),
      (7, "female", 27, 4, "no", 3, 17, 5, 5),
      (1, "male", 42, 15, "yes", 4, 16, 5, 5),
      (1, "female", 47, 15, "yes", 5, 14, 4, 5),
      (7, "female", 27, 4, "yes", 3, 18, 5, 4),
      (1, "female", 27, 7, "yes", 5, 14, 1, 4),
      (12, "male", 27, 1.5, "yes", 3, 17, 5, 4),
      (12, "female", 27, 7, "yes", 4, 14, 6, 2),
      (3, "female", 42, 15, "yes", 4, 16, 5, 4),
      (7, "female", 27, 10, "yes", 4, 12, 7, 3),
      (1, "male", 27, 1.5, "no", 2, 18, 5, 2),
      (1, "male", 32, 4, "no", 4, 20, 6, 4),
      (1, "female", 27, 7, "yes", 3, 14, 1, 3),
      (3, "female", 32, 10, "yes", 4, 14, 1, 4),
      (3, "male", 27, 4, "yes", 2, 18, 7, 2),
      (1, "female", 17.5, 0.75, "no", 5, 14, 4, 5),
      (1, "female", 32, 10, "yes", 4, 18, 1, 5),
      (7, "female", 32, 7, "yes", 2, 17, 6, 4),
      (7, "male", 37, 15, "yes", 2, 20, 6, 4),
      (7, "female", 37, 10, "no", 1, 20, 5, 3),
      (12, "female", 32, 10, "yes", 2, 16, 5, 5),
      (7, "male", 52, 15, "yes", 2, 20, 6, 4),
      (7, "female", 42, 15, "yes", 1, 12, 1, 3),
      (1, "male", 52, 15, "yes", 2, 20, 6, 3),
      (2, "male", 37, 15, "yes", 3, 18, 6, 5),
      (12, "female", 22, 4, "no", 3, 12, 3, 4),
      (12, "male", 27, 7, "yes", 1, 18, 6, 2),
      (1, "male", 27, 4, "yes", 3, 18, 5, 5),
      (12, "male", 47, 15, "yes", 4, 17, 6, 5),
      (12, "female", 42, 15, "yes", 4, 12, 1, 1),
      (7, "male", 27, 4, "no", 3, 14, 3, 4),
      (7, "female", 32, 7, "yes", 4, 18, 4, 5),
      (1, "male", 32, 0.417, "yes", 3, 12, 3, 4),
      (3, "male", 47, 15, "yes", 5, 16, 5, 4),
      (12, "male", 37, 15, "yes", 2, 20, 5, 4),
      (7, "male", 22, 4, "yes", 2, 17, 6, 4),
      (1, "male", 27, 4, "no", 2, 14, 4, 5),
      (7, "female", 52, 15, "yes", 5, 16, 1, 3),
      (1, "male", 27, 4, "no", 3, 14, 3, 3),
      (1, "female", 27, 10, "yes", 4, 16, 1, 4),
      (1, "male", 32, 7, "yes", 3, 14, 7, 4),
      (7, "male", 32, 7, "yes", 2, 18, 4, 1),
      (3, "male", 22, 1.5, "no", 1, 14, 3, 2),
      (7, "male", 22, 4, "yes", 3, 18, 6, 4),
      (7, "male", 42, 15, "yes", 4, 20, 6, 4),
      (2, "female", 57, 15, "yes", 1, 18, 5, 4),
      (7, "female", 32, 4, "yes", 3, 18, 5, 2),
      (1, "male", 27, 4, "yes", 1, 16, 4, 4),
      (7, "male", 32, 7, "yes", 4, 16, 1, 4),
      (2, "male", 57, 15, "yes", 1, 17, 4, 4),
      (7, "female", 42, 15, "yes", 4, 14, 5, 2),
      (7, "male", 37, 10, "yes", 1, 18, 5, 3),
      (3, "male", 42, 15, "yes", 3, 17, 6, 1),
      (1, "female", 52, 15, "yes", 3, 14, 4, 4),
      (2, "female", 27, 7, "yes", 3, 17, 5, 3),
      (12, "male", 32, 7, "yes", 2, 12, 4, 2),
      (1, "male", 22, 4, "no", 4, 14, 2, 5),
      (3, "male", 27, 7, "yes", 3, 18, 6, 4),
      (12, "female", 37, 15, "yes", 1, 18, 5, 5),
      (7, "female", 32, 15, "yes", 3, 17, 1, 3),
      (7, "female", 27, 7, "no", 2, 17, 5, 5),
      (1, "female", 32, 7, "yes", 3, 17, 5, 3),
      (1, "male", 32, 1.5, "yes", 2, 14, 2, 4),
      (12, "female", 42, 15, "yes", 4, 14, 1, 2),
      (7, "male", 32, 10, "yes", 3, 14, 5, 4),
      (7, "male", 37, 4, "yes", 1, 20, 6, 3),
      (1, "female", 27, 4, "yes", 2, 16, 5, 3),
      (12, "female", 42, 15, "yes", 3, 14, 4, 3),
      (1, "male", 27, 10, "yes", 5, 20, 6, 5),
      (12, "male", 37, 10, "yes", 2, 20, 6, 2),
      (12, "female", 27, 7, "yes", 1, 14, 3, 3),
      (3, "female", 27, 7, "yes", 4, 12, 1, 2),
      (3, "male", 32, 10, "yes", 2, 14, 4, 4),
      (12, "female", 17.5, 0.75, "yes", 2, 12, 1, 3),
      (12, "female", 32, 15, "yes", 3, 18, 5, 4),
      (2, "female", 22, 7, "no", 4, 14, 4, 3),
      (1, "male", 32, 7, "yes", 4, 20, 6, 5),
      (7, "male", 27, 4, "yes", 2, 18, 6, 2),
      (1, "female", 22, 1.5, "yes", 5, 14, 5, 3),
      (12, "female", 32, 15, "no", 3, 17, 5, 1),
      (12, "female", 42, 15, "yes", 2, 12, 1, 2),
      (7, "male", 42, 15, "yes", 3, 20, 5, 4),
      (12, "male", 32, 10, "no", 2, 18, 4, 2),
      (12, "female", 32, 15, "yes", 3, 9, 1, 1),
      (7, "male", 57, 15, "yes", 5, 20, 4, 5),
      (12, "male", 47, 15, "yes", 4, 20, 6, 4),
      (2, "female", 42, 15, "yes", 2, 17, 6, 3),
      (12, "male", 37, 15, "yes", 3, 17, 6, 3),
      (12, "male", 37, 15, "yes", 5, 17, 5, 2),
      (7, "male", 27, 10, "yes", 2, 20, 6, 4),
      (2, "male", 37, 15, "yes", 2, 16, 5, 4),
      (12, "female", 32, 15, "yes", 1, 14, 5, 2),
      (7, "male", 32, 10, "yes", 3, 17, 6, 3),
      (2, "male", 37, 15, "yes", 4, 18, 5, 1),
      (7, "female", 27, 1.5, "no", 2, 17, 5, 5),
      (3, "female", 47, 15, "yes", 2, 17, 5, 2),
      (12, "male", 37, 15, "yes", 2, 17, 5, 4),
      (12, "female", 27, 4, "no", 2, 14, 5, 5),
      (2, "female", 27, 10, "yes", 4, 14, 1, 5),
      (1, "female", 22, 4, "yes", 3, 16, 1, 3),
      (12, "male", 52, 7, "no", 4, 16, 5, 5),
      (2, "female", 27, 4, "yes", 1, 16, 3, 5),
      (7, "female", 37, 15, "yes", 2, 17, 6, 4),
      (2, "female", 27, 4, "no", 1, 17, 3, 1),
      (12, "female", 17.5, 0.75, "yes", 2, 12, 3, 5),
      (7, "female", 32, 15, "yes", 5, 18, 5, 4),
      (7, "female", 22, 4, "no", 1, 16, 3, 5),
      (2, "male", 32, 4, "yes", 4, 18, 6, 4),
      (1, "female", 22, 1.5, "yes", 3, 18, 5, 2),
      (3, "female", 42, 15, "yes", 2, 17, 5, 4),
      (1, "male", 32, 7, "yes", 4, 16, 4, 4),
      (12, "male", 37, 15, "no", 3, 14, 6, 2),
      (1, "male", 42, 15, "yes", 3, 16, 6, 3),
      (1, "male", 27, 4, "yes", 1, 18, 5, 4),
      (2, "male", 37, 15, "yes", 4, 20, 7, 3),
      (7, "male", 37, 15, "yes", 3, 20, 6, 4),
      (3, "male", 22, 1.5, "no", 2, 12, 3, 3),
      (3, "male", 32, 4, "yes", 3, 20, 6, 2),
      (2, "male", 32, 15, "yes", 5, 20, 6, 5),
      (12, "female", 52, 15, "yes", 1, 18, 5, 5),
      (12, "male", 47, 15, "no", 1, 18, 6, 5),
      (3, "female", 32, 15, "yes", 4, 16, 4, 4),
      (7, "female", 32, 15, "yes", 3, 14, 3, 2),
      (7, "female", 27, 7, "yes", 4, 16, 1, 2),
      (12, "male", 42, 15, "yes", 3, 18, 6, 2),
      (7, "female", 42, 15, "yes", 2, 14, 3, 2),
      (12, "male", 27, 7, "yes", 2, 17, 5, 4),
      (3, "male", 32, 10, "yes", 4, 14, 4, 3),
      (7, "male", 47, 15, "yes", 3, 16, 4, 2),
      (1, "male", 22, 1.5, "yes", 1, 12, 2, 5),
      (7, "female", 32, 10, "yes", 2, 18, 5, 4),
      (2, "male", 32, 10, "yes", 2, 17, 6, 5),
      (2, "male", 22, 7, "yes", 3, 18, 6, 2),
      (1, "female", 32, 15, "yes", 3, 14, 1, 5))

val colArray1: Array[String] = Array("affairs", "gender", "age", "yearsmarried", "children", "religiousness", "education", "occupation", "rating")

val data = dataList.toDF(colArray1:_*)

// ================（建立多层感知器分类器MLPC模型）================ //

data.createOrReplaceTempView("data")

// 字符类型转换成数值
val labelWhere = "case when affairs=0 then 0 else cast(1 as double) end as label"  // 这简直可以将多类别作为二分类的一种思路啊
val genderWhere = "case when gender='female' then 0 else cast(1 as double) end as gender"
val childrenWhere = "case when children='no' then 0 else cast(1 as double) end as children"

val dataLabelDF = spark.sql(s"select $labelWhere, $genderWhere,age,yearsmarried,$childrenWhere,religiousness,education,occupation,rating from data")

val featuresArray = Array("gender", "age", "yearsmarried", "children", "religiousness", "education", "occupation", "rating")

// 字段转换成特征向量
val assembler = new VectorAssembler().setInputCols(featuresArray).setOutputCol("features")
val vecDF: DataFrame = assembler.transform(dataLabelDF)
vecDF.show(10, truncate = false)

// 分割数据
val splits = vecDF.randomSplit(Array(0.6, 0.4), seed = 1234L)
val trainDF = splits(0)
val testDF = splits(1)

// 隐藏层结点数=2n+1，n为输入结点数
// 指定神经网络的图层：输入层8个节点(即8个特征)；两个隐藏层，隐藏结点数分别为9和8；输出层2个结点(即二分类)
val layers = Array[Int](8, 9, 8, 2)


// 建立多层感知器分类器MLPC模型
// 传统神经网络通常，层数<=5，隐藏层数<=3
// layers 指定神经网络的图层
// MaxIter 最大迭代次数
// stepSize 每次优化的迭代步长，仅适用于solver="gd"
// blockSize 用于在矩阵中堆叠输入数据的块大小以加速计算。 数据在分区内堆叠。 如果块大小大于分区中的剩余数据，则将其调整为该数据的大小。 建议大小介于10到1000之间。默认值：128
// initialWeights 模型的初始权重
// solver 算法优化。 支持的选项：“gd”（minibatch梯度下降）或“l-bfgs”。 默认值：“l-bfgs”
val trainer = new MultilayerPerceptronClassifier().setFeaturesCol("features").setLabelCol("label").setLayers(layers)
//.setMaxIter(100).setTol(1E-4).setSeed(1234L)
//.setBlockSize(128).setSolver("l-bfgs")
//.setInitialWeights(Vector).setStepSize(0.03)

// 训练模型
val model = trainer.fit(trainDF)
// 测试
val result = model.transform(testDF)
val predictionLabels = result.select("prediction", "label")

// 计算精度
val evaluator = new MulticlassClassificationEvaluator().setPredictionCol("prediction").setLabelCol("label").setMetricName("accuracy")
println("Accuracy: " + evaluator.evaluate(predictionLabels))




