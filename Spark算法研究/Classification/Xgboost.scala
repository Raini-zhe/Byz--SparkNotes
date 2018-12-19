
import org.apache.log4j.{ Level, Logger }
import org.apache.spark.{ SparkConf, SparkContext }
import ml.dmlc.xgboost4j.scala.spark.XGBoost
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.{ SparkSession, Row }
object Xgboost {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    val spark = SparkSession.builder.master("local").appName("example").
      config("spark.sql.warehouse.dir", s"file:///Users/shuubiasahi/Documents/spark-warehouse").
      config("spark.sql.shuffle.partitions", "20").getOrCreate()
    spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val path = "/Users/shuubiasahi/Documents/workspace/xgboost/demo/data/"
    val trainString = "agaricus.txt.train"
    val testString = "agaricus.txt.test"

    val train = spark.read.format("libsvm").load(path + trainString).toDF("label", "feature")

    val test = spark.read.format("libsvm").load(path + testString).toDF("label", "feature")

    val numRound = 15

    //"objective" -> "reg:linear", //定义学习任务及相应的学习目标
    //"eval_metric" -> "rmse", //校验数据所需要的评价指标  用于做回归

    val paramMap = List(
      "eta" -> 1f,
      "max_depth" -> 5, //数的最大深度。缺省值为6 ,取值范围为：[1,∞]
      "silent" -> 1, //取0时表示打印出运行时信息，取1时表示以缄默方式运行，不打印运行时信息。缺省值为0
      "objective" -> "binary:logistic", //定义学习任务及相应的学习目标
      "lambda" -> 2.5,
      "nthread" -> 1 //XGBoost运行时的线程数。缺省值是当前系统可以获得的最大线程数
    ).toMap
    val model = XGBoost.trainWithDataFrame(train, paramMap, numRound, 45, obj = null, eval = null, useExternalMemory = false, Float.NaN, "feature", "label")
    val predict = model.transform(test)

    val scoreAndLabels = predict.select(model.getPredictionCol, model.getLabelCol)
      .rdd
      .map { case Row(score: Double, label: Double) => (score, label) }

    //get the auc
    val metric = new BinaryClassificationMetrics(scoreAndLabels)
    val auc = metric.areaUnderROC()
    println("auc:" + auc)

  }

}

