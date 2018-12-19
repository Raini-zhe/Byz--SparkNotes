package web_classify

import org.apache.spark.sql.SparkSession
import org.apache.spark._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.reflect.ClassTag


object Test {

  def main(args: Array[String]): Unit = {

    Logger.getLongger("org.apache.spark").setLever(Lever.WARN)
    val spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("web_logisticRegression_classify")
      .getOrCreate()
    val sc = spark.sparkContext
    val sqlContext = spark.sqlContext
    //val training = spark.read.format("libsvm").load("file:///home/raini/spark/data/mllib/sample_libsvm_data.txt")


    /** =========================(属性图 属性图实例).========================*/
    // Create an RDD for the vertices
    val users: RDD[(VertexId, (String, String))] =  sc.parallelize(Array((3L, ("rxin", "student")), (7L, ("jgonzal", "postdoc")),
      (5L, ("franklin", "prof")), (2L, ("istoica", "prof"))))
    // Create an RDD for edges
    val relationships: RDD[Edge[String]] =
      sc.parallelize(Array(Edge(3L, 7L, "collab"), Edge(5L, 3L, "advisor"),
        Edge(2L, 5L, "colleague"), Edge(5L, 7L, "pi")))
    // 设置一个默认顶点0 的属性，去连接5->7, 形成一个回路图
    val defaultUser = ("John Doe", "Missing")
    // Build the initial Graph
    val graph = Graph(users, relationships, defaultUser)

    // (顶点属性的过滤)Count all users which are postdocs
    graph.vertices.filter { case (id, (name, pos)) => pos == "postdoc" }.count
    // (边属性的操作)Count all the edges where src > dst
    graph.edges.filter(e => e.srcId > e.dstId).count
    // 和上面表达一样
    graph.edges.filter { case Edge(src, dst, prop) => src > dst }.count

    // SQL的实现(Triplet)
    //SELECT src.id, dst.id, src.attr, e.attr, dst.attr FROM edges AS e LEFT JOIN vertices AS src, vertices AS dst ON e.srcId = src.Id AND e.dstId = dst.Id

    // EdgeTriplet类扩展了Edge类通过增加srcAttr 和dstAttr 成员，它们包含源和目的顶点属性。还有边的属性
    // 我们可以使用一个图的 triplet 视图来提供一些字符串描述用户之间的关系。
    // .srcAttr._1：源端属性第一个元素
    // .attr：边属性
    // .triplet.dstAttr._1：目的端属性第一个元素
    val facts: RDD[String] = graph.triplets.map(triplet =>
      triplet.srcAttr._1 + " is the " + triplet.attr + " of " + triplet.dstAttr._1)
    facts.collect.foreach(println(_))
    //      rxin is the collab of jgonzal
    //      franklin is the advisor of rxin
    //      istoica is the colleague of franklin


    /** =======================(图操作)==================== */

    graph.edges take 9
    // 计算每一个顶点的入度（目的顶点数，定义在GraphOps）- Use the implicit GraphOps.inDegrees operator
    val inDegrees: VertexRDD[Int] = graph.inDegrees
    // = Array((3,1), (7,2), (5,1))
    graph.outDegrees take 8
    graph.degrees.collect() // 所有 度（出入都算） 的个数
    graph.edges.collect()

    /** (操作列表概要)*/
    /** Summary of the functionality in the property graph */
    class Graph[VD, ED] {
      // Information about the Graph ===================================================================
      val numEdges: Long
      val numVertices: Long
      val inDegrees: VertexRDD[Int]
      val outDegrees: VertexRDD[Int]
      val degrees: VertexRDD[Int]
      // Views of the graph as collections =============================================================
      val vertices: VertexRDD[VD]
      val edges: EdgeRDD[ED]
      val triplets: RDD[EdgeTriplet[VD, ED]]

      // Functions for caching graphs ==================================================================
      def persist(newLevel: StorageLevel = StorageLevel.MEMORY_ONLY): Graph[VD, ED]

      def cache(): Graph[VD, ED]

      def unpersistVertices(blocking: Boolean = true): Graph[VD, ED]

      // Change the partitioning heuristic  ============================================================
      def partitionBy(partitionStrategy: PartitionStrategy): Graph[VD, ED]

      // Transform vertex and edge attributes ==========================================================
      def mapVertices[VD2](map: (VertexId, VD) => VD2): Graph[VD2, ED]

      def mapEdges[ED2](map: Edge[ED] => ED2): Graph[VD, ED2]

      def mapEdges[ED2](map: (PartitionID, Iterator[Edge[ED]]) => Iterator[ED2]): Graph[VD, ED2]

      def mapTriplets[ED2](map: EdgeTriplet[VD, ED] => ED2): Graph[VD, ED2]

      def mapTriplets[ED2](map: (PartitionID, Iterator[EdgeTriplet[VD, ED]]) => Iterator[ED2])
      : Graph[VD, ED2]

      // Modify the graph structure ====================================================================
      def reverse: Graph[VD, ED]

      def subgraph(// 子图
                   epred: EdgeTriplet[VD, ED] => Boolean = (x => true),
                   vpred: (VertexId, VD) => Boolean = ((v, d) => true))
      : Graph[VD, ED]

      def mask[VD2, ED2](other: Graph[VD2, ED2]): Graph[VD, ED]

      def groupEdges(merge: (ED, ED) => ED): Graph[VD, ED]

      // Join RDDs with the graph ======================================================================
      def joinVertices[U](table: RDD[(VertexId, U)])(mapFunc: (VertexId, VD, U) => VD): Graph[VD, ED]

      def outerJoinVertices[U, VD2](other: RDD[(VertexId, U)])
                                   (mapFunc: (VertexId, VD, Option[U]) => VD2)
      : Graph[VD2, ED]

      // Aggregate information about adjacent triplets =================================================
      def collectNeighborIds(edgeDirection: EdgeDirection): VertexRDD[Array[VertexId]]

      def collectNeighbors(edgeDirection: EdgeDirection): VertexRDD[Array[(VertexId, VD)]]

      def aggregateMessages[Msg: ClassTag](
                                            sendMsg: EdgeContext[VD, ED, Msg] => Unit,
                                            mergeMsg: (Msg, Msg) => Msg,
                                            tripletFields: TripletFields = TripletFields.All)
      : VertexRDD[A]

      // Iterative graph-parallel computation ==========================================================
      def pregel[A](initialMsg: A, maxIterations: Int, activeDirection: EdgeDirection)( // 后两个参数有默认值
        vprog: (VertexId, VD, A) => VD,
        sendMsg: EdgeTriplet[VD, ED] => Iterator[(VertexId, A)],
        mergeMsg: (A, A) => A)
      : Graph[VD, ED]

      // Basic graph algorithms ========================================================================
      def pageRank(tol: Double, resetProb: Double = 0.15): Graph[Double, Double]

      def connectedComponents(): Graph[VertexId, ED]

      def triangleCount(): Graph[Int, ED]

      def stronglyConnectedComponents(numIter: Int): Graph[VertexId, ED]
    }


    /** (属性操作) */
    class Graph[VD, ED] {
      // 都将返回一个新图
      def mapVertices[VD2](map: (VertexId, VD) => VD2): Graph[VD2, ED] // 修改顶点属性

      def mapEdges[ED2](map: Edge[ED] => ED2): Graph[VD, ED2] // 修改边属性

      def mapTriplets[ED2](map: EdgeTriplet[VD, ED] => ED2): Graph[VD, ED2] // 修改边属性（可同时调用顶点属性信息）
    }

    val newGraph = graph.mapVertices((id, attr) => mapUdf(id, attr))
    // 功能与上等同
    val newVertices = graph.vertices.map { case (id, attr) => (id, mapUdf(id, attr)) }
    //没有保存结构索引
    val newGraph = Graph(newVertices, graph.edges)

    //这些操作经常用来初始化图为了进行特殊计算或者排除不需要的属性。

    /** outerJoinVertices(给顶点加上一些属性信息)  */
    // 例如，给定一个图，它的出度作为顶点属性（给顶点加上一些属性信息），我们初始化它为PageRank：
    val inputGraph: Graph[Int, String] = graph.outerJoinVertices(graph.outDegrees)((vid, _, degOpt) => degOpt.getOrElse(0)) // 颗粒化，第二个参数相当于map
    graph.outerJoinVertices(graph.outDegrees)((vid, tri, degOpt) => (vid, tri, degOpt.getOrElse(0))).vertices.collect()
    // = Array((2,(2,(istoica,prof),1)), (3,(3,(rxin,student),1)), (7,(7,(jgonzal,postdoc),0)), (5,(5,(franklin,prof),2)))
    graph.outDegrees.collect()
    // = Array((2,1), (3,1), (5,2))  -- 7没有,所以配为0

    /** mapTriplets(修改边属性-顶点不变) */
    inputGraph.edges.collect()
    //  = Array(Edge(3,7,collab), Edge(5,3,advisor), Edge(2,5,colleague), Edge(5,7,pi))
    inputGraph.triplets.collect()
    //  = Array(((3,1),(7,0),collab), ((5,2),(3,1),advisor), ((2,1),(5,2),colleague), ((5,2),(7,0),pi))
    // Construct a graph where each edge contains the weight - = graph.triplets.map
    // and each vertex is the initial PageRank
    val outputGraph: Graph[Double, Double] = inputGraph.mapTriplets(triplet => 1.0 / triplet.srcAttr).mapVertices((id, _) => 1.0)
    inputGraph.mapTriplets(triplet => (triplet.srcAttr * 0.02, triplet.dstAttr * 100)).vertices.collect() // 顶点信息无变化
    // Array[(org.apache.spark.graphx.VertexId, Int)] = Array((2,1), (3,1), (7,0), (5,2))
    inputGraph.mapTriplets(triplet => (triplet.srcAttr * 0.02, triplet.dstAttr * 100)).edges.collect()
    // Edge[(Double, Int)]] = Array(Edge(3,7,(0.02,0)), Edge(5,3,(0.04,100)), Edge(2,5,(0.02,200)), Edge(5,7,(0.04,0)))
    inputGraph.triplets.map(t => (t.dstAttr*0.1,t.srcAttr*100)).collect()
    // Array[(Double, Int)] = Array((0.0,100), (0.1,200), (0.2,100), (0.0,200))
    inputGraph.mapTriplets(triplet => (triplet.srcAttr+"aa", triplet.attr)).edges.collect()
    //  = Array(Edge(3,7,(1aa,0)), Edge(5,3,(2aa,100)), Edge(2,5,(1aa,200)), Edge(5,7,(2aa,0)))

    /** mapEdges(修改边属性) */
    inputGraph.mapEdges(e=> e.attr+"---").edges.collect()


    /** ==================(结构操作).**********************************************
      * */
    // 当前，GraphX仅仅支持一个简单的常用结构操作，将来会不断完善。
    class Graph[VD, ED] {
      def reverse: Graph[VD, ED]
      // reverse操作返回一个新图，其所有的边方向反向。有时这是有用的，例如，尝试计算反转的PageRank。因为反转操作没有修改顶点或者边属性或者改变边数量，这能够高效的实现没有数据移动或者复制。

      def subgraph(epred: EdgeTriplet[VD, ED] => Boolean,
                   vpred: (VertexId, VD) => Boolean): Graph[VD, ED]
      // subgraph操作利用顶点和边判断，返回图包含满足判断的顶点，满足边判断的顶点，满足顶点判断的连接顶点。subgraph 操作可以用在一些情景，限制感兴趣的图顶点和边，删除损坏连接。

      def mask[VD2, ED2](other: Graph[VD2, ED2]): Graph[VD, ED]
      //  mask操作通过subgraph的返回构建一个子图，其包含顶点和边也被构建在输入图中。

      def groupEdges(merge: (ED, ED) => ED): Graph[VD, ED]
      // // groupEdges操作合并了多重图的并行边（例如，顶点之间的重复边）-(就是边去重呗)
    }

    /** 移除损坏连接-例子 */
    // subgraph ~ map ： 针对顶点的操作
    val validGraph = graph.subgraph(vpred = (id, attr) => attr._2 != "postdoc")
    // 移除-顶点-属性= "prof" 的 users链接
    validGraph.edges.collect.foreach(println(_))
    validGraph.triplets.map(
      triplet => triplet.srcAttr._1 + " is the " + triplet.attr + " of " + triplet.dstAttr._1
    ).collect.foreach(println(_))

    /** mask操作通过subgraph的返回构建一个子图，其包含顶点和边也被构建在输入图中。 */
    // Run Connected Components
    val ccGraph = graph.connectedComponents() // 连通图
    ccGraph.edges.collect()
    val validCCGraph = ccGraph.mask(validGraph) // 构建这个子图
    validCCGraph.vertices.collect()
    validCCGraph.edges.collect()

    // groupEdges操作合并了多重图的并行边（例如，顶点之间的重复边）。在一些数字应用程序中，并行边能被增加（权重融合）到一个边，因此减少了图的大小。
    graph.groupEdges((a,b)=>b).edges.collect()


    /** =====================(join操作)***************************
      *
      * */
    // 在很多情况下，需要将外部数据集合（RDDs）添加到图中。
    // 例如，我们可能有额外的用户属性，我们想把它融合到一个存在图中或者我们可能想拉数据属性从一个图到另一个图。这些任务可以使用join操作来实现。下面我们列出了关键的join操作：
    class Graph[VD, ED] {
      def joinVertices[U](table: RDD[(VertexId, U)])(map: (VertexId, VD, U) => VD) // 函数普通参数写法
      : Graph[VD, ED]

      def outerJoinVertices[U, VD2](table: RDD[(VertexId, U)])(map: (VertexId, VD, Option[U]) => VD2) // 颗粒化写法
      : Graph[VD2, ED]
      // outerJoinVertices(给顶点加上一些属性信息)
    }
    // ·joinVertices操作·: 连接vertices 和输入RDD，返回一个新图，其顶点属性通过应用用户定义map函数到joined vertices结果上获得的。在RDD顶点没有一个匹配值保留其原始值。
    // 注意：(保持顶点唯一性，如下)如果RDD对一个给定顶点包含超过一个值，仅仅有一个将会使用。因此，建议输入RDD保持唯一性，这可以使用下面方法，预索引结果值，加快join执行速度。
    val nonUnique: RDD[(VertexId, Double)] = graph.vertices.map(t=>(t._1,t._1.toDouble))
    val uniqueCosts: VertexRDD[Double] = graph.vertices.aggregateUsingIndex(nonUnique, (a, b) => a + b)
    uniqueCosts.collect()
    val joinedGraph = graph.joinVertices(uniqueCosts)((id, oldCost, extraCost) => oldCost + extraCost )//
    joinedGraph.edges.collect
    joinedGraph.vertices.collect
    // [outerJoinVertices]: 除了定义的map函数可被应用到所有顶点和可以改变顶点类型外，outerJoinVertices和joinVertices相似。
    // 因为在输入的RDD中不是所有的顶点都有一个匹配值，map函数使用了一个Option类型，我们将没有值的顶点属性置为0。
    // 例如，我们可以设置一个图对PageRank通过初始化顶点属性使用出度：
    val outDegrees: VertexRDD[Int] = graph.outDegrees
    val degreeGraph = graph.outerJoinVertices(outDegrees) { (id, oldAttr, outDegOpt) => // 颗粒化写法，f(a)(b)，(b)依赖于(a)
      outDegOpt match {
        case Some(outDeg) => outDeg
        case None => 0 // No outDegree means zero outDegree
      }
    }
    degreeGraph.edges.collect()
    // [柯里函数]模式的多参数列表（例如f(a)(b)）被使用在上面的实例中。当我将f(a)(b) 写为 f(a,b)，将意味着类型接口b将不会依赖于a。
    // 因此用户需要提供类型注释对用户自定义函数：
    val joinedGraph = graph.joinVertices(uniqueCosts,
      (id: VertexID, oldCost: Double, extraCost: Double) => oldCost + extraCost)


    /** ===============相邻聚合（Neighborhood Aggregation) ================ */

    // 在图分析任务中一个关键步骤就是聚集每一个顶点的邻居信息。
    // 例如，我们想知道每一个用户的追随者数量或者追随者的平均年龄。一些迭代的图算法（像PageRank,最短路径和联通组件）反复的聚集相邻顶点的属性（像当前pagerank值，源的最短路径，最小可到达的顶点id）。
    // graph.mapReduceTriplets 已改为新的graph.AggregateMessages。
    /** ---信息聚集（Aggregate Messages (aggregateMessages)）------ */
    // 在GraphX中核心的聚集操作是aggregateMessages。
    // 这个操作应用了一个用户定义的sendMsg函数到图中的每一个边 triplet，然后用mergeMsg函数在目的节点聚集这些信息。
    class Graph[VD, ED] {
      def aggregateMessages[Msg: ClassTag](
                                            sendMsg: EdgeContext[VD, ED, Msg] => Unit, //sendMsg 作为map-reduce中的map函数
                                            mergeMsg: (Msg, Msg) => Msg, // 用户定义的mergeMsg函数使用到相同顶点的两个信息，将它们计算产出一条信息。作为reduce函数
                                            tripletFields: TripletFields = TripletFields.All) // 其表明在EdgeContext中什么数据可以被访问（例如，有源顶点属性没有目的顶点属性）。tripletsFields 可能的选项被定义在TripletsFields中，默认值为 TripletFields.All，其表明用户定义的sendMsg 函数可以访问EdgeContext的任何属性。
      : VertexRDD[Msg] // aggregateMessages函数返回一个VertexRDD[Msg]，其包含了到达每一个顶点的融合信息（Msg类型）。没有接收一个信息的顶点不被包含在返回的VertexRDD中。
    }

    // 例如
    // ，如果我们计算每一个用户追随者的平均年龄，我们仅仅要求源属性即可，所以我们使用 TripletFields.Src 来表明我们仅仅使用源属性。
    import org.apache.spark.graphx.{Graph, VertexRDD}
    import org.apache.spark.graphx.util.GraphGenerators
    // Create a graph with "age" as the vertex property.
    // Here we use a random graph for simplicity.// 生成一个对数正态图
    val graph1: Graph[Double, Int] = GraphGenerators.logNormalGraph(sc, numVertices = 100).mapVertices((id, _) => id.toDouble)
    // Compute the number of older followers and their total age
    val olderFollowers: VertexRDD[(Int, Double)] = graph1.aggregateMessages[(Int, Double)](
      triplet => {
        // Map Function
        if (triplet.srcAttr > triplet.dstAttr) { // 保证数据是同一方向
          // 发送 计数器和年龄 消息到目标顶点 - Send message to destination vertex containing counter and age
          triplet.sendToDst(1, triplet.srcAttr) // 源顶点信息 (map-类似wordCount)
        } // 得到每个顶点追随者的数量
      },
      // Add counter and age
      (a, b) => (a._1 + b._1, a._2 + b._2) // Reduce Function -- 聚合上面得到的（数量，年龄）
    )
    olderFollowers.sortBy(_._1).collect.foreach(println(_))

    /** 比上面好理解 - 类似groupByKey 多了计数 */
    // 可用于电话诈骗中
    graph.edges.collect()
    val grou = graph.aggregateMessages[(Int, String)](// 返回类型：[(Int, String)]
      triplet => {
        // Map Function
        //if (triplet.srcAttr > triplet.dstAttr) {
        triplet.sendToSrc(1, triplet.srcAttr._1) // 源顶点信息
        //} // 得到每个顶点追随者的数量
      },
      // Add counter and age
      (a, b) => (a._1 + b._1, a._2 + "," + b._2) // Reduce Function -- 聚合上面得到的（数量，年龄）
    )
    grou.sortBy(_._1).collect.foreach(println(_))
    // (2,(1,istoica))
    // (3,(1,rxin))
    // (5,(2,franklin,franklin)) <- 节点5出现了2次



    /** --- 计算度（Degree）信息----------
      *
      * 一个普通的聚合任务是计算每一个顶点的度：每一个顶点边的数量。在有向图的情况下，它经常知道入度，出度和每个顶点的总度。
      * GraphOps 类包含了每一个顶点的一系列的度的计算。
      * 例如：在下面将计算最大入度，出度和总度： */
    graph.edges.collect.foreach(println(_))

    // Define a reduce operation to compute the highest degree vertex
    def max(a: (VertexId, Int), b: (VertexId, Int)): (VertexId, Int) = {
      if (a._2 > b._2) a else b
    }

    // Compute the max degrees
    val maxInDegree: (VertexId, Int) = graph.inDegrees.reduce(max)
    // 返回(7,2)，即（顶点,个数）
    val maxOutDegree: (VertexId, Int) = graph.outDegrees.reduce(max)
    val maxDegrees: (VertexId, Int) = graph.degrees.reduce(max) // 统计出度+入度所有顶点的个数（扁平化累计）


    /** ----- 邻居收集------------
      *
      * 在一些情形下，通过收集每一个顶点的邻居顶点和它的属性来表达计算是更加容易的。
      * 通过使用 collectNeighborIds 和 collectNeighbors 操作。
      *
      * 注：这些操作代价比较高，由于复制信息和要求大量的通信。尽可能直接使用aggregateMessages 操作完成相同的计算。 */

    class GraphOps[VD, ED] {
      def collectNeighborIds(edgeDirection: EdgeDirection): VertexRDD[Array[VertexId]] // 需要指定方向（出度还是入度）

      def collectNeighbors(edgeDirection: EdgeDirection): VertexRDD[Array[(VertexId, VD)]]
    }
    // 相当于groupBy(出度节点), 得到（出度,(入度集合)）
    graph.collectNeighborIds(EdgeDirection.Out).collect()
    // 相当于groupBy(入度节点), 得到（入度,(出度集合)）
    graph.collectNeighborIds(EdgeDirection.In).collect
    graph.collectNeighborIds(EdgeDirection.Either).collect.foreach(n => println(n._1 + "'s neighbors : " + n._2.distinct.mkString(",")))

    graph.collectNeighbors(EdgeDirection.In).collect.foreach(n => println(n._1 + "'s in neighbors : " + n._2.mkString(",")))
    graph.collectNeighbors(EdgeDirection.Out).collect.foreach(n => println(n._1 + "'s out neighbors : " + n._2.mkString(",")))
    graph.collectNeighbors(EdgeDirection.Either).collect.foreach(n => println(n._1 + "'s neighbors : " + n._2.distinct.mkString(",")))


    /** ----- 缓存和取消缓存------------
      *
      * 当使用一个图多次时，调用Graph.cache()。
      * 默认，缓存的RDDs和图将会保留在内存中直到内存不足，迫使它们以LRU顺序被驱除。对于迭代计算，从过去相关迭代产生的中间结果将被缓存，即使最终被驱除，不需要的数据存储在内存中将会减缓垃圾回收。取消不需要的中间结果的缓存将会更加高效。这涉及每次迭代物化（缓存和强迫）一个图和RDD，取消所有其他数据集缓存，仅仅使用物化数据集在将来迭代中。然而，因为图由多个RDDs组成，正确解除他们的持久化是比较难的。对迭代计算我们推荐使用 Pregel API，其能正确的解除中间结果的持久化。 */
    graph.cache()

    /** ----- Pregel API------------
      *
      * 在GraphX中，更高级的Pregel操作是一个约束到图拓扑的批量同步（bulk-synchronous）并行消息抽象。
      * Pregel操作执行一系列高级步骤，顶点从过去的超级步骤接收他们流入信息总和，对顶点属性计算一个新值，发送信息到邻居节点在下一个高级步骤。
      * 不像Pregel，信息作为边triplet函数被平行计算，信息计算访问源和目的顶点属性。没有接收信息的顶点在一个高级步骤中被跳过。
      * 当没有保留信息时，pregel终止迭代并返回最终图。
      *
      * 注意： Pregel使用两个参数列表（像graph.pregel(list1)(list2)）。
      * 第一个参数列表包含配置参数包括初始化信息，最大迭代次数和发送信息边方向（默认沿着out边）。
      * 第二个参数列表包含用户自定义函数，对应接收信息（顶点程序Vprog），计算信息（sendMsg）和组合信息（mergeMsg）。
      *
      * 一个典型的Pregel计算过程如下：读取输入，初始化该图，当图被初始化好后，运行一系列的supersteps，每一次superstep都在全局的角度上独立运行，直到整个计算结束，输出结果。 */
    class GraphOps[VD, ED] {
      def pregel[A](initialMsg: A, maxIter: Int = Int.MaxValue, activeDir: EdgeDirection = EdgeDirection.Out)
                   (vprog: (VertexId, VD, A) => VD, // 接收信息（顶点程序Vprog）
                    sendMsg: EdgeTriplet[VD, ED] => Iterator[(VertexId, A)], // 计算信息（sendMsg）和组合信息（mergeMsg）
                    mergeMsg: (A, A) => A)
      : Graph[VD, ED] = {
        // 第一次迭代，对每个节点用vprog函数计算。 在每个顶点接收初始消息
        var g = mapVertices((vid, vdata) => vprog(vid, vdata, initialMsg)).cache()
        // 根据发送、聚合信息的函数计算下次迭代用的信息。
        var messages = g.mapReduceTriplets(sendMsg, mergeMsg)
        // 数一下还有多少节点活跃
        var activeMessages = messages.count()
        // 实现循环直到没有消息保持或maxiterations
        var i = 0
        while (activeMessages > 0 && i < maxIterations) {
          // 接收消息并更新顶点 - Receive the messages and update the vertices.
          g = g.joinVertices(messages)(vprog).cache()
          // .cache()
          val oldMessages = messages
          // 发送新消息，跳过双方都没有收到消息的边。
          messages = g.mapReduceTriplets(
            sendMsg, mergeMsg, Some((oldMessages, activeDirection))).cache()
          activeMessages = messages.count()
          i += 1
        }
        g
      }
    }

    /** 使用Pregel操作表达计算，像下面的单元最短路径实例 */
    import org.apache.spark.graphx.{Graph, VertexId}
    import org.apache.spark.graphx.util.GraphGenerators

    // 含边属性的图 - A graph with edge attributes containing distances
    val graph2: Graph[Long, Double] = GraphGenerators.logNormalGraph(sc, numVertices = 100).mapEdges(e => e.attr.toDouble)
    graph2.edges.collect()
    val sourceId: VertexId = 42
    // 源点 - The ultimate source
    // (1)初始化图，除了根的所有顶点都有距离无穷大。 - 首先将所有除了源顶点的其它顶点的属性值设置为无穷大，源顶点的属性值设置为0.
    val initialGraph1 = graph2.mapVertices(
      (id, _) =>
        if (id == sourceId) 0.0 else Double.PositiveInfinity)
    initialGraph.edges.collect()
    // (2)Superstep 0：然后对所有顶点用initialmsg进行初始化，实际上这次初始化并没有改变什么。
    val sssp = initialGraph1.pregel(Double.PositiveInfinity)(
      (id, dist, newDist) => math.min(dist, newDist), // Vertex Program 这个方法的作用是更新节点VertexId的属性值为新值，以利于innerJoin操作
      triplet => {
        // Send Message - map函数
        if (triplet.srcAttr + triplet.attr < triplet.dstAttr) {
          Iterator((triplet.dstId, triplet.srcAttr + triplet.attr)) // 其中id为接受消息方
        } else {
          Iterator.empty
        }
      },
      (a, b) => math.min(a, b) // Merge Message - reduce函数
    )
    println(sssp.vertices.collect.mkString("\n"))

    //输出结果，注意pregel返回的是更新VertexId属性的graph，而不是VertexRDD[(VertexId,VD)]
    //找出路径最短的点
    def min(a: (VertexId, Double), b: (VertexId, Double)): (VertexId, Double) = {
      if (a._2 < b._2) a else b
    }

    print("最短节点：" + sssp.vertices.filter(_._1 != 0).reduce(min)); //注意过滤掉源节点


    /** (pregel实验) */
    // 初始化图，除了根的所有顶点都有距离无穷大
    val initialGraph = graph.mapVertices(
      (id, _) =>
        if (id == sourceId) 0.0 else Double.PositiveInfinity)
    initialGraph.vertices.collect()
    // Array[(org.apache.spark.graphx.VertexId, Double)] = Array((4,Infinity), (0,Infinity), (2,Infinity), (3,Infinity), (7,Infinity), (5,Infinity))
    // Array[org.apache.spark.graphx.Edge[String]] = Array(Edge(2,5,colleague), Edge(3,7,collab), Edge(5,3,advisor), Edge(4,0,student), Edge(5,0,colleague), Edge(5,7,pi))

    val sssp = initialGraph.pregel(Double.PositiveInfinity)(
      (id, dist, newDist) => math.min(dist, newDist), // 这个方法的作用是更新节点VertexId的属性值为新值，以利于innerJoin操作
      edge => {
        // Send Message
        if (edge.srcAttr < edge.dstAttr) {
          Iterator((edge.dstId, edge.dstId)) // .srcAttr
        } else {
          Iterator.empty
        }
      },
      (a, b) => math.min(a, b) // Merge Message
    )
    println(sssp.vertices.collect.mkString("\n"))
    println(sssp.edges.collect.mkString("\n"))
    graph.vertices.collect.mkString("\n")
    print("最短节点：" + sssp.vertices.filter(_._1 != 0).reduce(min)); //注意过滤掉源节点


    /** ===================== 图算法=====================
      * 算法被包含在org.apache.spark.graphx.lib包里面，能被Graph通过GraphOps直接访问。 */

    /** ----- PageRank ------------
      *
      * 该PageRank模型提供了两种调用方式：
      * 第一种：（静态）在调用时提供一个参数number，用于指定迭代次数，即无论结果如何，该算法在迭代number次后停止计算，返回图结果。
      * 第二种：（动态）在调用时提供一个参数tol，用于指定前后两次迭代的结果差值应小于tol，以达到最终收敛的效果时才停止计算，返回图结果。
      */
    // Load the edges as a graph
    //val graph = GraphLoader.edgeListFile(sc, "graphx/data/followers.txt")
    // Run PageRank
    val ranks = graph.pageRank(0.0001).vertices
    // 第二种
    val rank2 = graph.staticPageRank(100)
    // Join the ranks with the usernames
    val users = sc.textFile("graphx/data/users.txt").map { line =>
      val fields = line.split(",")
      (fields(0).toLong, fields(1))
    }
    val ranksByUsername = users.join(ranks).map {
      case (id, (username, rank)) => (username, rank)
    }
    // Print the result
    println(ranksByUsername.collect().mkString("\n"))


    /** ----- Connected Components (联通图-聚类)------------
      *
      * // Edge(1,2,1)
      * // Edge(2,3,1)
      * // Edge(3,1,1)
      * // Edge(4,5,1)
      * // Edge(5,6,1)
      * // Edge(6,7,1)
      * // 1,2,3相连，4,5,6,7相连
      *
      * 连通图算法使用最小编号的顶点标记图的连通体 - 连通图以图中最小Id作为label给图中顶点打属性
      * 例如，在一个社会网络，连通图近似聚类。
      * GraphX 在 ConnectedComponents 对象中包含一个算法实现，我们计算连通图实例，数据集和 PageRank部分一样： */

    import org.apache.spark.graphx.GraphLoader

    // Load the graph as in the PageRank example
    val graph3 = GraphLoader.edgeListFile(sc, "file:///home/raini/spark/data/graphx/followers.txt")
    graph3.edges.collect()
    // Find the connected components
    val cc = graph3.connectedComponents()
    //.vertices
    // Join the connected components with the usernames
    val users = sc.textFile("file:///home/raini/spark/data/graphx/users.txt").map { line =>
      val fields = line.split(",")
      (fields(0).toLong, fields(1))
    }
    val ccByUsername = users.join(cc.vertices).map {
      case (id, (username, cc)) => (id, username, cc)
    }
    // Print the result
    println(ccByUsername.collect().mkString("\n"))

    // 取出id为2的顶点的label
    val cc_label_of_vid_2: Long = cc.vertices.filter { case (id, label) => id == 2 }.first._2
    println(cc_label_of_vid_2)
    // 1

    // 取出相同类标的顶点
    val vertices_connected_with_vid_2: RDD[(Long, Long)] = cc.vertices.filter { case (id, label) => label == cc_label_of_vid_2 }
    vertices_connected_with_vid_2.collect.foreach(println(_))
    //(4,1)
    //(2,1)
    //(1,1)


    val vids_connected_with_vid_2: RDD[Long] = vertices_connected_with_vid_2.map(v => v._1)
    vids_connected_with_vid_2.collect.foreach(println(_))
    // 4
    // 2
    // 1

    // 取出子图
    val graph_include_vid_2 = graph.subgraph(vpred = (vid, attr) => vids_list.contains(vid))
    graph_include_vid_2.vertices.collect.foreach(println(_))
    // (2,1)
    // (1,1)
    // (3,1)


    /** ----- Triangle Counting------------
      *
      * 当顶点有两个邻接顶点并且它们之间有边相连，它就是三角形的一部分。
      * GraphX 在 TriangleCount对象中实现了一个三角形计数算法，其确定通过每一个顶点的三角形数量，提供了一个集群的测量。
      * 我们计算社交网络三角形的数量，数据集同样使用PageRank部分数据集。
      * 注意：三角形数量要求边是标准方向（srcId < dstId），图使用Graph.partitionBy进行分区。 */

    import org.apache.spark.graphx.{GraphLoader, PartitionStrategy}

    val triCounts = graph.triangleCount().vertices
    triCounts.collect()
    // Join the triangle counts with the usernames
    val users = sc.textFile("data/graphx/users.txt").map { line =>
      val fields = line.split(",")
      (fields(0).toLong, fields(1))
    }
    val triCountByUsername = users.join(triCounts).map { case (id, (username, tc)) =>
      (username, tc)
    }
    // Print the result
    println(triCountByUsername.collect().mkString("\n"))


    /** ----- Examples ------------ */

    import org.apache.spark.graphx.GraphLoader

    // Load my user data and parse into tuples of user id and attribute list
    val users = sc.textFile("file:///home/raini/spark/data/graphx/users.txt").map(line => line.split(",")).map(parts => (parts.head.toLong, parts.tail))

    // Parse the edge data which is already in userId -> userId format
    val followerGraph = GraphLoader.edgeListFile(sc, "file:///home/raini/spark/data/graphx/followers.txt")
    followerGraph.edges.collect()

    // Attach the user attributes
    val graph4 = followerGraph.outerJoinVertices(users) {
      case (uid, deg, Some(attrList)) => attrList
      // Some users may not have attributes so we set them as empty
      case (uid, deg, None) => Array.empty[String]
    }
    graph4.edges.collect()
    graph4.vertices.collect()

    // Restrict the graph to users with usernames and names
    val subgraph = graph.subgraph(vpred = (vid, attr) => attr.size == 2)

    // Compute the PageRank
    val pagerankGraph = subgraph.pageRank(0.001)

    // Get the attributes of the top pagerank users
    val userInfoWithPageRank = subgraph.outerJoinVertices(pagerankGraph.vertices) {
      case (uid, attrList, Some(pr)) => (pr, attrList.toList)
      case (uid, attrList, None) => (0.0, attrList.toList)
    }

    println(userInfoWithPageRank.vertices.top(5)(Ordering.by(_._2._1)).mkString("\n"))


  }
}


ED：边RDD
VD：顶点RDD

存储：把边分割进行分别存储，顶点也一样（如：1个源顶点有6个目的顶点，分三份，每份存2个）
  策略：EdgePatition2D、EdgePatition1D、RandomVetexCut、CanonicalRandomVertexCut

函数里的>：
  if (triplet.srcAttr > triplet.dstAttr) { // 保证数据是同一方向


GraphX创建图并可视化的核心技术:
  http://blog.csdn.net/wzwdcld/article/details/51585093

Spark GraphX 对图进行可视化:
  http://blog.csdn.net/simple_the_best/article/details/75097615
  http://www.demodashi.com/demo/10644.html














//
