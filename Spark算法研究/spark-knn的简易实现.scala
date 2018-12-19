spark-knn，spark是一个很优秀的分布式计算框架，本文实现的knn是基于欧几里得距离公式实现的，下面开始起简单的实现，可能有多问题希望大家能够给指出来。

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    val conf = new SparkConf( ).setAppName("knn")
    conf.set("spark.serializer","org.apache.spark.serializer.KryoSerializer")
    val sc = new SparkContext( conf )

    val k:Int = 6
    val path = "hdfs://master:9000/knn.txt"
    val data = sc.textFile( path ).map( line =>{
      val pair = line.split( "\\s+" )
      ( pair( 0 ).toDouble,pair( 1 ).toDouble ,pair( 2 ) )
    } )
    val total:Array[ RDD[(Double,Double,String)] ] = data.randomSplit(Array( 0.7,0.3 ) )
    val train = total( 0 ).cache()
    val test = total( 1 ).cache()
    train.count()
    test.count()
    val bcTrainSet = sc.broadcast( train.collect() )

    val bck = sc.broadcast( k )

    val resultSet = test.map{ line => {
      val x = line._1
      val y = line._2
      val trainDatas = bcTrainSet.value
      val set = scala.collection.mutable.ArrayBuffer.empty[(Double, String)]
      trainDatas.foreach( e => {
        val tx = e._1.toDouble
        val ty = e._2.toDouble
        val distance = Math.sqrt( Math.pow( x - tx, 2 ) + Math.pow( y - ty, 2 ) )
        set.+= (( distance, e._3 ) )
      })
      val list = set.sortBy( _._1 )
      val categoryCountMap = scala.collection.mutable.Map.empty[String, Int]
      val k = bck.value
      for ( i <- 0 until k ){
        val category = list(i)._2
        val count = categoryCountMap.getOrElse( category, 0 ) + 1
        categoryCountMap += ( category -> count )
      }
      val ( rCategory,frequency ) = categoryCountMap.maxBy( _._2 )
      ( x, y, rCategory )
    }}

    resultSet.repartition(1).saveAsTextFile( "hdfs://master:9000/knn/result" )


以上实现是最简单的实现方式。可以采用加权方法，例如在统计次数的时候使用距离的倒数乘以次数作为最终的次数










#######---------------------2：

 //获取样本数据
    val rawData = sc.textFile("D:/logdata/kmeans.txt")
    //将样本数据转化为模型可操作的向量集
    val labelAndData = rawData.map { line =>
      val buffer = line.split(',').toBuffer
      val label = buffer.remove(0)
      val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
      (label, vector)
    }
    //将样本数据向量集缓存
    val data = labelAndData.values.cache()
    //建立Kmeans学习模型
    val kmeans = new KMeans()
    kmeans.setK(3)
    //训练数据
    val model = kmeans.run(data)
    //打印簇心点
    model.clusterCenters.foreach(println)


    //欧氏距离的计算函数
    def distance(a: Vector, b: Vector): Double = {
      math.sqrt(a.toArray.zip(b.toArray).map(p => p._1 - p._2).map(d => d * d).sum)
    }
    //计算向量到模型簇心点的距离
    def distToCentroid(datum: Vector, model: KMeansModel) = {
      val cluster = model.predict(datum)
      val centroid = model.clusterCenters(cluster)
      distance(centroid, datum)
    }
    //计算所有点到簇心点的距离集合
    val distances = data.map(datum =>
      distToCentroid(datum, model)
    )
    //获取最大的第五个值为阈值
    val threshold = distances.top(5).last

    //测试数据获取
    val testRawData = sc.textFile("D:/logdata/kmeans")
    val testLabelAndData = testRawData.map { line =>
      val buffer = line.split(',').toBuffer
      val label = buffer.remove(0)
      val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
      (label, vector)
    }
    //将测试数据集缓存
    val testData = testLabelAndData.values.cache()

    //异常数据集过滤并打印结果
    val anomalies=testData.filter { x =>
      distToCentroid(x, model) > threshold
    }.collect().foreach(println)



