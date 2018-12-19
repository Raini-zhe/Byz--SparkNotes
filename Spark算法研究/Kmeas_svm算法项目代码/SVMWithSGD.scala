package web_classify

/**
  *
  * spark-shell --master spark://biyuzhe:7077 --executor-memory 3G --driver-memory 2G
  * 
  * 参数： "file:///home/raini/data/train_noheader.tsv"（文件路径）
         // iter：迭代次数，step：步长，reg：平滑，updater: 正则化参数
  *
  * */

import org.apache.spark.mllib.classification.{ClassificationModel, SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.{L1Updater, SimpleUpdater, SquaredL2Updater, Updater}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.{SparkConf, SparkContext}

object SVMWithSGD {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("web_classify")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .getOrCreate()
    //spark.sparkContext.getConf.registerKryoClasses(Array(classOf[dataSetClass]))
    val sc = spark.sparkContext
    val sqlContext = spark.sqlContext

    val rawData = sc.textFile(args(0))
    // url	urlid	boilerplate(title引用)	 alchemy_category（网站类别）	alchemy_category_score(类别分数,权重)	avglinksize(平均连接数)	commonlinkratio_1	commonlinkratio_2 ...	label

    val records = rawData.map(line => line.split("\t"))
    records.first()
    //res0: Array[String] = Array("http://www.bloomberg.com/news/2010-12-23/ibm-predicts-holographic-calls-air-breathing-batteries-by-2015.html", "4042", "{""title"":""IBM Sees Holographic Calls Air Breathing Batteries ibm sees holographic calls, air-breathing batteries"",""body"":""A sign stands outside the International Business Machines Corp IBM Almaden Research Center campus in San Jose California Photographer Tony Avelar Bloomberg Buildings stand at the International Business Machines Corp IBM Almaden Research Center campus in the Santa Teresa Hills of San Jose California Photographer Tony Avelar Bloomberg By 2015 your mobile phone will project a 3 D image of anyone who calls and your laptop will be powered by kinetic energy At least that s what International Business Machines Corp sees ...


    /** 执行函数 */
    // 1. 切分 训练集/测试集
    val Array(trainData, testData) = dataProcessing(sc, sqlContext, rawData).
      randomSplit(Array(0.7,0.3),123)

    trainData.cache()
    testData.cache()

    /* -----------------交叉验证---------------- */

    //加快多次模型训练速度, 缓存标准化后的数据
    //scaledDataCats.cache()
    val numIteration = 100

    //1迭代
    val iterRasults = Seq(1, 5, 10, 50).map { param =>
      val model = trainWithParams(trainData, 0.0, param, new SimpleUpdater, 1.0)
      creatMetrics(s"$param iterations", trainData, model)
    }
    iterRasults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%")}
    //1 iterations, AUC = 64.95%
    //    5 iterations, AUC = 66.62%
    //    10 iterations, AUC = 66.55%
    //    50 iterations, AUC = 66.81%


    //2步长 大步长收敛快，太大可能导致收敛到局部最优解
    val stepResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = trainWithParams(trainData, 0.0, numIteration, new SimpleUpdater, param)
      creatMetrics(s"$param step size", trainData, model)
    }
    stepResults.foreach { case (param, auc) => println(f"$param,AUC = ${auc * 100}%2.2f%%")
    }
    //    0.001 step size,AUC = 64.97%
    //      0.01 step size,AUC = 64.96%
    //      0.1 step size,AUC = 65.52%
    //      1.0 step size,AUC = 66.55%
    //      10.0 step size,AUC = 61.92%


    //3正则化, new L1Updater , new SimpleUpdater
    val regResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map{ param =>
      val model = trainWithParams(trainData, param, numIteration, new SquaredL2Updater, 1.0)
      creatMetrics(s"${param} L2 regularization parameter",trainData, model)
    }
    regResults.foreach{ case (param,auc) => println(f"$param,AUC = ${auc * 100}%2.2f%%")
    }
    //    0.001 L2 regularization parameter,AUC = 66.55%
    //      0.01 L2 regularization parameter,AUC = 66.55%
    //      0.1 L2 regularization parameter,AUC = 66.63%
    //      1.0 L2 regularization parameter,AUC = 66.04%
    //      10.0 L2 regularization parameter,AUC = 35.33%

    /* -----------------取最好的参数 训练模型、预测数据---------------- */

    val iter = 100
    val step = 1.0
    val reg = 1.0
    val updater: Updater = new L1Updater

    val svmModel: SVMModel = trainSVMModel(trainData, iter, step, reg, updater)
    svmModel.clearThreshold()

    val scoreAndLabels = testData.map { point =>
      val score = svmModel.predict(point.features)
      (score, point.label)
    }

    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()


    trainData.unpersist()
    testData.unpersist()
  }



  def dataProcessing(sc: SparkContext, sqlContext: SQLContext, rawData: RDD[String]): RDD[LabeledPoint] = {

    val records = rawData.map(line => line.split("\t"))
    /** 数据清理： 用0替换缺失值？，去掉多余”*/
    val lv = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"",""))
      val lable = trimmed(r.size - 1).toInt

      val features = trimmed.slice(4,r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
      LabeledPoint(lable, Vectors.dense(features))
    }
    // 标准化数据
    val scalerCats = new StandardScaler(withMean = true, withStd = true).fit(lv.map(lp => lp.features))
    val scaledDataCats = lv.map(
      lp =>
      LabeledPoint(lp.label, scalerCats.transform(lp.features))
    )
    scaledDataCats
  }

  def trainSVMModel(data: RDD[LabeledPoint],iter: Int =100, step:Double, reg:Double, updater: Updater): SVMModel = {
    val svmAlg = new SVMWithSGD()
    svmAlg.optimizer
      .setNumIterations(iter)
      .setRegParam(0.1)
      .setUpdater(updater)
      .setStepSize(step)
    val modelL1 = svmAlg.run(data)
    modelL1
  }

  //定义辅助函数，根据给定输入训练模型 (输入数据， 正则化参数， 迭代次数， 正则化形式， 步长)
  def trainWithParams(input: RDD[LabeledPoint], regParam: Double, numIntrations: Int, updater: Updater, stepSize: Double) = {
    val svmAlg = new SVMWithSGD()
    svmAlg.optimizer
      .setNumIterations(numIntrations)
      .setUpdater(updater)
      .setRegParam(regParam)
      .setStepSize(stepSize)
    svmAlg.run(input)
  }
  //定义第二个辅助函数,根据输入数据和分类模型 计算AUC
  def creatMetrics(label: String, data: RDD[LabeledPoint], model: ClassificationModel) = {
    val scoreAndLabels = data.map { point =>
      (model.predict(point.features),point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (label, metrics.areaUnderROC())
  }

}
