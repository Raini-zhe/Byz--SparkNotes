//聚类 幂迭代聚类(PIC)
/**
 * 快速迭代聚类是一种简单可扩展的图聚类方法
 */






    val data = sc.textFile("C:\\coding\\jars\\spark-2.2.0-bin-hadoop2.7\\data\\mllib\\pic_test_data.txt")
    val similarities: RDD[(Long, Long, Double)] = data.map { line =>
      val parts = line.split(' ')
      (parts(0).toLong, parts(1).toLong, parts(2).toDouble) //（源顶点, 目的顶点, 边的相似度）
    }

    // 使用快速迭代算法将数据聚类
    val pic = new PowerIterationClustering()
      .setK(5)  //k : 期望聚类数
      .setInitializationMode("degree") //模型初始化，默认使用”random” ，即使用随机向量作为初始聚类的边界点，可以设置”degree”（就是图论中的度）
      //随机初始化后，特征值为随机值；度初始化后，特征为度的平均值。
      //度向量会给图中度大的节点分配更多的初始化权重，使其值可以更平均和快速的分布，从而更快的局部收敛。
      .setMaxIterations(20) //幂迭代最大次数
    val model = pic.run(similarities)


    //打印出所有的簇
    val res = model.assignments.collect().map(x=>x.id -> x.cluster).groupBy[Int](_._2).map(x=>x._1 -> x._2.map(x=>x._1).mkString(","))
    res.foreach(x=>println("cluster "+x._1+": " +x._2))



//-------------(将聚类得到的图重新做成图

    val vertexRDD = model.assignments.map{f=>(f.id, f.cluster)}
    val clusterGraph = Graph(vertexRDD, inputGraph.edges)
    clusterGraph








