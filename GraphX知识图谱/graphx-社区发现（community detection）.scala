LPA算法（标签传播算法）
1、为所有的节点指定一个唯一的标签
2、逐轮刷新所有节点的标签，直到达到收敛要求为止，对于每一轮刷新，节点标签刷新的规则如下：
对某一个节点，考察其所有邻居节点的标签，并进行统计，将出现个数最多的那个标签赋给当前节点，当个数最多的标签不唯一时，随机选择一个。
目前spark-graphx实现了该社区发现算法。该算法优点就是算法原理简单，实现简便，廉价的计算;缺点就是该算法完成的社区发现存在结果震荡的问题。

SLPA算法
是一种重叠社区发现算法，其中涉及到一个重要的阈值参数r，通过r的适当选取，可将其退化为非重叠模型。
SLPA引入了Listener和Speaker两个比较形象的概念，你可以这么理解，在刷新节点标签的过程中，任意选取一个节点作为listener，则其所有邻居节点就是他的speaker了，speaker通常不止一个，一大群speaker在七嘴八舌时，listener到底该听谁的呢？我们可以通过一定的规则来解决这个问题。例如LPA中出现次数最多的标签，这也是一个规则。
SLPA最大的特点在于，它会记录每一个节点在刷新迭代过程中的历史标签序列，（例如迭代T次，则每个节点保存一个长度为T的序列）当迭代停止后，对于每一个节点历史标签序列中（互异）标签出现的频率做统计，按照某一给定的阈值过滤掉那些出现频率小的标签，
剩下即为该节点的标签了（可能会有很多个）
object LabelPropagation {
/**
* 在社交网络中用lpa发现社区
*网络中的每个节点都有一个初始的社区标识，每次节点发送自己的社区标识到自己所有的邻居节点并且更新所有节点的模态社区
* lpa是一个标准的社区发现算法，它的计算是廉价的，虽然他不能保证一定会收敛，但是它可以使用某些规则结束其迭代
*
* @tparam ED 边属性的类型
* @param graph 需要计算社区的图
* @param maxSteps 由于是静态实现，设置最大的迭代次数
* @return 返回的结果在图的点属性中包含社区的标识 */
def run[VD, ED: ClassTag](graph: Graph[VD, ED], maxSteps: Int): Graph[VertexId, ED] = {
//初始化每个定点的社区表示为当前结点的id值
val lpaGraph = graph.mapVertices { case (vid, _) => vid }
//定义消息的发送函数，将顶点的社区标识发送到相邻顶点
def sendMessage(e: EdgeTriplet[VertexId, ED]): Iterator[(VertexId, Map[VertexId, Long])] = {
Iterator((e.srcId, Map(e.dstAttr -> 1L)), (e.dstId, Map(e.srcAttr -> 1L)))
}
//顶点的消息聚合函数 将每个节点消息聚合，做累加；例如一个定点出现了两次，id->2
def mergeMessage(count1: Map[VertexId, Long], count2: Map[VertexId, Long])
: Map[VertexId, Long] = {
(count1.keySet ++ count2.keySet).map { i =>
val count1Val = count1.getOrElse(i, 0L)
val count2Val = count2.getOrElse(i, 0L)
i -> (count1Val + count2Val)
}.toMap
}
//该函数用于在完成一次迭代的时候，将第一次的结果和原图做关联
如果当前结点的message为空，则将该节点的社区标识设置为当前结点的id，如果不为空，在根据其他节点在出现的次数求最大值，（可以把他看成是一种规则，slpa是基于重叠社区的发现，slpa则使用出现次数的阈值等规则）
def vertexProgram(vid: VertexId, attr: Long, message: Map[VertexId, Long]): VertexId = {
if (message.isEmpty) attr else message.maxBy(_._2)._1
}
val initialMessage = MapVertexId, Long
Pregel(lpaGraph, initialMessage, maxIterations = maxSteps)(
vprog = vertexProgram,
sendMsg = sendMessage,
mergeMsg = mergeMessage)
}
}
Logger.getLogger(“org.apache.spark”).setLevel(Level.WARN)
Logger.getLogger(“org.eclipse.jetty.server”).setLevel(Level.OFF)
val conf = new SparkConf().setAppName(“lpa”).setMaster(“local”)
val sc = new SparkContext( conf )
val rdd = sc.makeRDD(Array(
“1 2”,”1 3”,”2 4”,”3 4”,”3 5”,”4 5”,”5 6”,”6 7”,
“6 9”,”7 11”,”7 8”,”9 8”,”9 13”,”8 10”,”10 13”,
“13 12”,”10 11”,”11 12”
))
val edge =rdd .map( line =>{
val pair = line.split(“\s+”)
Edge( pair(0).toLong,pair(1).toLong,1L )
})
val graph = Graph.fromEdges( edge,0 )
val label = LabelPropagation.run(graph ,5 )

label.vertices.foreach( println )
