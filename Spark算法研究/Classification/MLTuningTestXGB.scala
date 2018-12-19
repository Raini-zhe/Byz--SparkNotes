

object MLTuningTestXGB {

  val sparkSession = SparkSession.builder().master("local").appName("testApp").getOrCreate()
  val sparkCtx = sparkSession.sparkContext

  def crossValidationTest(): Unit = {
    //prepare training data
    val training = sparkSession.createDataFrame(Seq(
      (0L, "a b c d e spark", 1.0),
      (1L, "b d", 0.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 0.0),
      (4L, "b spark who", 1.0),
      (5L, "g d a y", 0.0),
      (6L, "spark fly", 1.0),
      (7L, "was mapreduce", 0.0),
      (8L, "e spark program", 1.0),
      (9L, "a e c l", 0.0),
      (10L, "spark compile", 1.0),
      (11L, "hadoop software", 0.0)
    )).toDF("id", "text", "label")

    //prepare testing data
    val test = sparkSession.createDataFrame(Seq(
      (12L, "spark i j k"),
      (13L, "l m n"),
      (14L, "mapreduce spark"),
      (15L, "apache hadoop")
    )).toDF("id", "text")

    //set the pipeline
    val tokenizer = new Tokenizer()
    .setInputCol("text")
    .setOutputCol("words")

    val hashingTF = new HashingTF()
    .setInputCol(tokenizer.getOutputCol)
    .setOutputCol("features")

    val paramMap = List(
      "num_rounds" -> 100,
      "eta" -> 0.1,
      "max_depth" -> 6,
      "silent" -> 1,
      "ntreelimit" -> 1000,
      "objective" -> "binary:logistic",
      "subsample" -> 0.8,
      "nworkers" -> 1
    ).toMap

    val xgbEstimator = new XGBoostEstimator(paramMap)

    val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, xgbEstimator))

    //now create a paramGrid for estimator
    val paramGrid = new ParamGridBuilder()
      .addGrid(xgbEstimator.round, Array(20, 50))
      .addGrid(xgbEstimator.eta, Array(0.1, 0.4))
      .build()

    //then wrap the pipeline in a crossValidator instance
    val crossValidator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)

    //run cross validation, get the model
    val cvModel = crossValidator.fit(training)

    //get the prediction
    cvModel.transform(test).show()
  }

}

