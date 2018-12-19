package cluster

// 参数：文件路径  （其它参数已经在代码里默认，可设置成自定义）

import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd._
import org.apache.spark.{SparkConf, SparkContext}

object NetworkTrafficAnomalyDetection_KMeans {

  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setAppName("K-means"))
    val rawData = sc.textFile(args(0),1)

    // 阈值 - 该(100)中心点之后的都是异常点
    val threshold_farClusterCentre_point = 100
    val K = 50
    /**---------------------------( 使用距离公式来 确定K值 )--------------------------*/
    clusteringTake1(rawData)
    /**--------------------------( 得到 k=20， 做聚类 )----------------------------------*/
    val c2 = clusteringTake2(rawData)
    val c21 = c2.map{
      case (clu, data)=>
        "cluster" + clu + ":\n" + data.mkString("\n")
    }//.collect()
    //c21.coalesce(1).saveAsTextFile("file:///home/raini/data/Spark_Advanced_Data_Analysis/Chapter 5/clusteringTake2_result_test")

    /**---------------(优化：利用熵的标号信息选取K值)-(略显多余，效果和上面差不多)-（跳过该步骤）-------------------------*/
    val c3 = clusteringTake3(rawData)

    /**--------------------------(实现-异常检测)----------------------------------*/
    val anomal = anomalies(rawData)
    anomal.take(10).foreach(println)
    anomal.saveAsTextFile("file:///home/raini/data/Spark_Advanced_Data_Analysis/Chapter 5/result_test")

  }

  /* 1.(数据准备) */
  def buildCategoricalAndLabelFunction(rawData: RDD[String]): (String => (String,Vector)) = {
    val splitData = rawData.map(_.trim.split(','))
    val protocols = splitData.map(_(1)).distinct().collect().zipWithIndex.toMap
    val services = splitData.map(_(2)).distinct().collect().zipWithIndex.toMap
    val tcpStates = splitData.map(_(3)).distinct().collect().zipWithIndex.toMap
    (line: String) => {
      val buffer = line.split(',').toBuffer
      val protocol = buffer.remove(1)
      val service = buffer.remove(1)
      val tcpState = buffer.remove(1)
      val label = buffer.remove(buffer.length - 1)
      val vector = buffer.map(_.toDouble)

      val newProtocolFeatures = new Array[Double](protocols.size)
      newProtocolFeatures(protocols(protocol)) = 1.0
      val newServiceFeatures = new Array[Double](services.size)
      newServiceFeatures(services(service)) = 1.0
      val newTcpStateFeatures = new Array[Double](tcpStates.size)
      newTcpStateFeatures(tcpStates(tcpState)) = 1.0

      vector.insertAll(1, newTcpStateFeatures)
      vector.insertAll(1, newServiceFeatures)
      vector.insertAll(1, newProtocolFeatures)

      (label, Vectors.dense(vector.toArray))
    }
  }
  /* 2.（正则化） */
  def buildNormalizationFunction(vectorData: RDD[Vector]): RDD[Vector] = {
    println("====buildNormalizationFunction====")
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(vectorData)
    scaler.transform(vectorData)
  }

  // 偶四距离
  def distance(a: Vector, b: Vector) = {
    Math.sqrt(a.toArray.zip(b.toArray).map(p => p._1 - p._2).map(d => d*d).sum)
  }
  // 数据到质心的距离
  def distToCentroid(datum: Vector, model: KMeansModel) = {
    val cluster = model.predict(datum) // 类簇
    val centroid = model.clusterCenters(cluster) // 类簇的质心
    //model.computeCost(RDD) // no等于下面（使用误差平方之和来评估数据模型）
    distance(centroid, datum)
  }

  // 为一个给定 k 值的模型定义平均质心距离函数
  def clusteringScore(data: RDD[Vector], k: Int): Double = {
    val kmeans = new KMeans()
    kmeans.setK(k)
    val model = kmeans.run(data)
    data.map(datum => distToCentroid(datum, model)).mean()
  }
  def clusteringScore11(data: RDD[Vector], k: Int): Double = {
    val kmeans = new KMeans()
    kmeans.setK(k)
    val model = kmeans.run(data)
    model.computeCost(data)
  }
  // 同上
  def clusteringScore2(data: RDD[Vector], k: Int): Double = {
    val kmeans = new KMeans()
    kmeans.setK(k)
    kmeans.setEpsilon(1.0e-6)//设置一个阈值,该阈值控制聚类过程中簇质心进行有效移动的最小值。降低该阈值能使质心继续移动更长的时间。➊默认为 1.0e-4,这里比默认值小。
    val model = kmeans.run(data)
    data.map(datum => distToCentroid(datum, model)).mean()
  }


  /**-----------------------------( 使用距离公式来 确定K值 )--------------------------*/
  def clusteringTake1(rawData: RDD[String]): Unit = {

    val parseFunction = buildCategoricalAndLabelFunction(rawData)
    // 便于说明,把原始数据和解析后的特征向量放在一起
    val originalAndData = rawData.map(line => (line, parseFunction(line)._2))
    val vectorData = originalAndData.values
    // 正则化
    val normalizedData = buildNormalizationFunction(vectorData)

    (5 to 30 by 5).map(k => (k, clusteringScore(normalizedData, k))).
      foreach(println)
    (5 to 30 by 5).map(k => (k, clusteringScore11(normalizedData, k))).
      foreach(println)
    /*[.par.] Scala的并行集合(parallel collection)。这样对每个 k 值的聚类计算可以在Spark shell 中并行执行.
              能提高集群的总体吞吐率。当然,这里有一个临界点,同时提交的任务数超过这个临界点后吞吐率反而会下降。.*/
    (5 to 30 by 5).par.map(k => (k, clusteringScore2(normalizedData, k))).toList.
      foreach(println)

    normalizedData.unpersist()
  }

  /**--------------------------(得到 k=20， 做聚类)----------------------------------*/

  def clusteringTake2(rawData: RDD[String]): RDD[(Int, Iterable[String])] = {
    val parseFunction = buildCategoricalAndLabelFunction(rawData)
    // 便于说明,把原始数据和解析后的特征向量放在一起
    val originalAndData = rawData.map(line => (line, parseFunction(line)._2))
    val vectorData = originalAndData.values
    // 正则化
    val normalizedData = buildNormalizationFunction(vectorData).cache()

    val kmeans = new KMeans().setK(20).setMaxIterations(60).setEpsilon(1.0e-6)
    val model = kmeans.run(normalizedData)

    val pred = model.predict(normalizedData) zip originalAndData.keys
    val result = pred.groupByKey().mapValues( line => line.take(50))

    normalizedData.unpersist()
    result
  }

  def clusterDataNum(normalizedLabelsAndData: RDD[(String,Vector)],k:Int) = {

    val kmeans = new KMeans().setK(k).setMaxIterations(60).setEpsilon(1.0e-6)
    val model = kmeans.run(normalizedLabelsAndData.values)

    // ➊ 对每个数据预测簇类别 -> （标签，类）
    val labelsAndClusters = normalizedLabelsAndData.mapValues(model.predict)
    // ➋ 对换键和值 -> (类，标签)
    val clustersAndLabels = labelsAndClusters.map(_.swap)
    // ➌ 按簇提取标号集合 -> （标签集）
    val labelsInCluster = clustersAndLabels.groupByKey().values
    // ➍ 计算集合中各簇标号出现的次数 -> （类，（标签，次数））
    val labelCounts = labelsInCluster.map(_.groupBy(l => l).map(_._2.size))

    println("labelCounts:"+labelCounts.collect().foreach(println))
    normalizedLabelsAndData.unpersist()
  }



  /**---------------(优化：利用熵的标号信息选取K值)-(略显多余，效果和上面差不多)-（跳过该步骤）-------------------------*/

  def clusteringTake3(rawData: RDD[String]): Unit = {
    val parseFunction = buildCategoricalAndLabelFunction(rawData)
    val vectorData = rawData.map(parseFunction).values
    val vectorData_keys = rawData.map(parseFunction).keys
    // 正则化
    val normalizedData = buildNormalizationFunction(vectorData).cache()

    val k_v = vectorData_keys zip normalizedData

    (1 to 6 by 2).map(k =>
      (k, clusteringScore3(k_v, k))).toList.foreach(println)

    normalizedData.unpersist()
  }

  def entropy(counts: Iterable[Int]) = {
    val values = counts.filter(_ > 0)
    val n: Double = values.sum
    values.map { v =>
      val p = v / n
      -p * math.log(p)
    }.sum
  }

  def clusteringScore3(normalizedLabelsAndData: RDD[(String,Vector)],k:Int) = {

    val kmeans = new KMeans().setK(k).setMaxIterations(60).setEpsilon(1.0e-6)
    val model = kmeans.run(normalizedLabelsAndData.values)

    // ➊ 对每个数据预测簇类别 -> （标签，类）
    val labelsAndClusters = normalizedLabelsAndData.mapValues(model.predict)

    // ➋ 对换键和值 -> (类，标签)
    val clustersAndLabels = labelsAndClusters.map(_.swap)

    // ➌ 按簇提取标号集合 -> （标签集）
    val labelsInCluster = clustersAndLabels.groupByKey().values

    // ➍ 计算集合中各簇标号出现的次数 -> （类，（标签，次数））
    val labelCounts = labelsInCluster.map(_.groupBy(l => l).map(_._2.size))

    // ➎ 根据簇大小计算熵的加权平均
    val n = normalizedLabelsAndData.count()

    val entro = labelCounts.map(m => m.sum * entropy(m)).sum / n

    println("labelCounts:"+labelCounts.collect().foreach(println))
    normalizedLabelsAndData.unpersist()
    entro
  }

  /**--------------------------(实现-异常检测)----------------------------------*/
  // 度量新数据点到最近的簇质心的距离。如果这个距离超过某个阈值,那么就表示这个新数据点是异常的。
  def buildAnomalyDetector(normalizedData: RDD[Vector]): (Vector => Boolean) = {
    normalizedData.cache()
    val kmeans = new KMeans()
    kmeans.setK(20).setMaxIterations(80)
    kmeans.setEpsilon(1.0e-6)
    val model = kmeans.run(normalizedData)
    normalizedData.unpersist()

    // 设置阈值，距离中心元的 - 默认第100个点为阈值
    val distances = normalizedData.map(datum => distToCentroid(datum, model))
    val threshold = distances.top(380).last

    // 异常点
    (datum: Vector) => distToCentroid(datum, model) > threshold
  }

  def anomalies(rawData: RDD[String]): RDD[String] = {
    val parseFunction = buildCategoricalAndLabelFunction(rawData)
    // 便于说明,我们把原始数据和解析后的特征向量放在一起
    val originalAndData = rawData.map(line => (line, parseFunction(line)._2))
    val vectorData = originalAndData.values
    //println("val originalAndData = rawData.map(line => (line, parseFunction(line)._2))"+vectorData.take(3))
    // 正则化
    val normalizedData = buildNormalizationFunction(vectorData)
    // 数据到质心的距离 - 建立异常检测
    val anomalyDetectorFunction = buildAnomalyDetector(normalizedData)
    //
    val anomalies = originalAndData.filter {
      case (original, datum) => anomalyDetectorFunction(datum)
    }.keys
    anomalies//.take(10).foreach(println)
  }

}

//anomalies(rawData)
