import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.{MultilayerPerceptronClassificationModel, MultilayerPerceptronClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
/**
  * Created by daniel on 3/23/17.
  */
object FlightDelayClassifier {
  val spark = SparkSession
    .builder
    .appName("FlightDelayClassifier")
    .config("spark.executor.memory", "4g")
    .config("spark.master", "local")
    .getOrCreate()
  val modelName = "FlightDelayClassifier"

  def trainModel(dataPath:String): (MultilayerPerceptronClassificationModel, Dataset[Row], Dataset[Row]) ={
    // Load the data stored in LIBSVM format as a DataFrame.
    val data = spark.read.format("libsvm")
      .load(dataPath)

    // Split the data into train and test
    val splits = data.randomSplit(Array(0.7, 0.3), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

    val layers = Array[Int](4, 5, 4, 3)

    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1525L)
      .setMaxIter(3)
    val model = trainer.fit(train)
    return (model,train,test)
  }

  def saveModel(model:MultilayerPerceptronClassificationModel ): String ={
    val df = DateTimeFormatter.ofPattern("<MM-dd-yyy>hh-mm-ss")
    val now = ZonedDateTime.now()
    model.save("../model/"+modelName+df.format(now)+".model")
    return modelName+df.format(now)
  }

  def findAccuracy(model:MultilayerPerceptronClassificationModel,test:Dataset[Row]):(MulticlassClassificationEvaluator,Dataset[Row])={
    // compute accuracy on the test set
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    return (evaluator,predictionAndLabels)
  }

  def loadModel(name:String):MultilayerPerceptronClassificationModel={
    return MultilayerPerceptronClassificationModel.load("../model/"+name+".model")
  }

  def useModel(model: MultilayerPerceptronClassificationModel, dataSet: Dataset[Row]): DataFrame ={
    val result = model.transform(dataSet)
    val predictionAndfeatures = result.select("prediction", "features")
    return predictionAndfeatures
  }

  def getData(path:String):(Dataset[Row],Dataset[Row])={
    val data = spark.read.format("libsvm")
      .load(path)

    // Split the data into train and test
    val splits = data.randomSplit(Array(0.7, 0.3), seed = 1234L)
    val train = splits(0)
    val test = splits(1)
    return (train,test)
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    val (model,train,test) = trainModel("../data/modeldata_W/DOT_2008_W.libsvm")
    val name = saveModel(model)
    val model1 = loadModel(name)
    //val (train,test) = getData("../data/modeldata_W/DOT_2008_W.libsvm")
    val (evaluator,predictionAndLabels) = findAccuracy(model,test)
    println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels))

    val predictionResults = useModel(model1,test.select("features"))

    println(predictionResults.show(3))
    spark.stop()
  }
}


// 交叉验证用（对数据比较了解好像也没有必要）：
// 
// val algorithmName = "Multi-Layer Perceptron Classifier"
//
// def generate(data: Dataset[Observation], target: String): CrossValidatorModel = {
//     val mpc = new MultilayerPerceptronClassifier()
//     val paramGrid = getTuningParams(mpc)
//     val model = generateModel(data, target, mpc, algorithmName, Some(paramGrid))
//     return model
// }
//
// def getTuningParams(mpc: MultilayerPerceptronClassifier): ParamGridBuilder = {
//     val paramGrid = new ParamGridBuilder()
//         .addGrid(mpc.maxIter, Array(10, 30, 60))
//         .addGrid(mpc.layers, Array(
//             Array[Int](4, 15, 2),
//             Array[Int](4, 10, 2),
//             Array[Int](4, 3, 2),
//             Array[Int](4, 7, 2)))
//     return paramGrid
// }
