package graph

import breeze.numerics.{abs, sqrt}
import org.apache.commons.net.daytime.DaytimeTCPClient
import org.apache.spark.graphx._
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkContext, SparkConf}
import scala.util.Random
import org.apache.spark.rdd.RDD
/**
 * Created by sara on 9/30/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object Spectral {

  def dotGraph(g1 : Graph[Double, Double], g2 : Graph[Double, Double]) : Double = {
    val dotGraph = g1.joinVertices(g2.vertices) { case (id, f1, f2) => f1 * f2}
    val dot = dotGraph.vertices.reduce{case ((v1,f1),(v2,f2)) => if (v1 > v2) {(v1, f1+f2)} else {(v2, f1+f2)}}._2
    return dot
  }

  def normVertices(vertices: VertexRDD[Double]): Double = {
    val squared = vertices.mapValues({( (id,f1)=>f1*f1)})
    val norm = squared.reduce{case ((v1,f1),(v2,f2)) => if (v1 > v2) {(v1, f1+f2)} else {(v2, f1+f2)}}._2
    return sqrt(norm)
  }

  def normGraph(g1: Graph[Double, Double]): Double = {
    return normVertices(g1.vertices)
  }

  def sumVertices(vertices: VertexRDD[Double]): Double = {
    val sum = vertices.reduce{case ((v1,f1),(v2,f2)) => if (v1 > v2) {(v1, f1+f2)} else {(v2, f1+f2)}}._2
    return sum
  }

  def normalize(graph: Graph[Double, Double]) : Graph[Double,Double] = {
    val s = normVertices(graph.vertices);
    val ngraph =  graph.mapVertices((id, v )=> (v/s));
    return ngraph
  }

  def normalizeV(v: VertexRDD[Double]) : VertexRDD[Double] = {
    val s = normVertices(v);
    val nv =  v.mapValues((id, v )=> (v/s));
    return nv
  }


  //def subtractGraph(g1: Graph[Double,Double], g2: Graph[Double, Double])

  def gram_schmidt(g1: Graph[Double, Double], g2: Graph[Double, Double]): Graph[Double, Double] = {
    val tupleGraph = g1.mapVertices{case (id, v1) => (v1, 0.0)} //Graph(g1.vertices.map{case (id,v1)=> (id,(v1,0.0))}, g1.edges)
    val joinGraph = tupleGraph.joinVertices(g2.vertices) { case (id, (f1,_), f2) => (f1,f2)}
    val dotValue = joinGraph.vertices.map{ case (id,(f1,f2))=> f1*f2}.reduce((f1,f2)=>f1 + f2);
    //val norm = g1.vertices.map{case (id, f)=> f*f}.reduce((f1,f2)=> f1+f2)
    //val norm= dotGraph(g1, g1)
    //val dot_g1_g2 = dotGraph(g1, g2)
    val norm = 1.0
    val coef = dotValue / norm
    //val orthogonal_g2 = g1.joinVertices(g2.vertices) { case (id, v1, v2) => v2 - (v1 * coef )}
    val orthogonal_g2 = joinGraph.mapVertices{ case (id, (v1, v2)) => v2 - (v1 * coef )}
    //val norm_g2 = normalize(orthogonal_g2)
    //val norm_g1 = normalize(g1)

    orthogonal_g2
  }

  def main(args: Array[String]): Unit = {
    Random.setSeed(System.currentTimeMillis());
    val filename = args(0);

    val conf = new SparkConf().setAppName("spectral");

    //conf.set("spark.akka.frameSize", "30");
    val sc = new SparkContext(conf)
    val inputgraph = GraphLoader.edgeListFile(sc,filename)
    val numNode = inputgraph.numVertices.toInt;
    println(numNode)
    //val output = inputgraph.collectNeighborIds(EdgeDirection.Out)
    val degreeRDD = inputgraph.outDegrees
    //val edgeRDD = sc.parallelize(inputgraph.edges.collect(),8)
    //val degreeGraph = Graph(degreeRDD, inputgraph.edges);
    //val edgeRDD2 = edgeRDD.map(e=>(e.srcId,e.dstId))
    //val graph1=Graph.fromEdgeTuples(edgeRDD2, 0, Some(PartitionStrategy.RandomVertexCut))
   // val graph2 = Graph(degreeRDD, edgeRDD);
    //val floattriplet = sc.parallelize[EdgeTriplet[Int,Int]](degreeGraph.triplets.collect() );
    //val floatgraph = floattriplet.
    val agraph = Graph(degreeRDD, inputgraph.edges)

    //val pgraph =agraph.mapTriplets({(triplet=> 1.0 / triplet.srcAttr.toDouble)})
    val pgraph =agraph.mapTriplets{triplet=> 1.0 / (sqrt(triplet.srcAttr) * sqrt(triplet.dstAttr))}


    println("Creating v1:")
    var pgraph_v1 = pgraph.mapVertices[Double]( (vertexId, a)=> Random.nextDouble()).partitionBy(PartitionStrategy.RandomVertexCut)
    println("Creating v2:")
    var pgraph_v2 = pgraph_v1.mapVertices{case (id, f1)=> Random.nextDouble()};
    //var pgraph_v2 = pgraph.mapVertices[Double]( (vertexId, a)=> Random.nextDouble()).partitionBy(PartitionStrategy.RandomVertexCut)
    //val norm_v1 = normGraph(pgraph_v1)
    pgraph_v1.cache();
    pgraph_v2.cache();
    inputgraph.unpersist();
    //pgraph.cache()

    //pgraph_v1.vertices.foreach{ case (vertexId, v)=> println(vertexId.toString + ":" + v)}
    //val randoms = Seq.fill(numNode)(Random.nextInt)
    //pgraph.edges.foreach{case e => println( e.srcId.toString + " " + e.dstId.toString + ":" + e.attr )}
    //val directedTopoGraph = pgraph.subgraph{case e=> (e.srcId > e.dstId)}
    //pgraph = normalize(pgraph);
    //val sum = sumVertices(pgraph.vertices);
    //var ngraph = pgraph;
    //var nv = pgraph.vertices;
    var error = 10.0
    var last_iter = 0

    //val dot = dotGraph(pgraph_v1, pgraph_v2)
    //val norm = normalize()
    println("nomalize v1")
    pgraph_v1 = normalize(pgraph_v1)

    println("Run gram-schmidth for random vectors")
    val opgraph_v2 = gram_schmidt(pgraph_v1, pgraph_v2)
    //pgraph_v1 = npgraph_v1;
    //pgraph_v2 = npgraph_v2;
    //val dot = dotGraph(pgraph_v1, pgraph_v2);
    //println("dot = " + dot)

    println("normalize v2");
    pgraph_v2 = normalize(pgraph_v2)
    val gap = new Array[Double](numNode)
    for (i <-0 to 30 if error > 0.0001) {
      println(" =============== iteration: " + i + " ====================")
      println("Compute P*v1");
      val v1_vertices = pgraph_v1.aggregateMessages[Double](
        sendMsg = ctx => ctx.sendToSrc(ctx.attr * ctx.dstAttr),
        mergeMsg = _ + _,
        TripletFields.Dst).cache()

      println("Compute P*v2")
      val v2_vertices = pgraph_v2.aggregateMessages[Double](
        sendMsg = ctx => ctx.sendToSrc(ctx.attr * ctx.dstAttr),
        mergeMsg = _ + _,
        TripletFields.Dst).cache()

      println("Normalize v1")
      val npgraph_v1 = normalize(Graph(v1_vertices, pgraph_v1.edges))

      println("Run gram-schmidt for v2")
      //val opgraph_v2 = gram_schmidt(Graph(v1_vertices, pgraph_v1.edges), Graph(v2_vertices, pgraph_v2.edges))
      val opgraph_v2 = gram_schmidt(npgraph_v1, Graph(v2_vertices, pgraph_v2.edges))


      println("Normalize v2")
      val npgraph_v2 = normalize(opgraph_v2)
      //val sum = sumVertices(v);
      //val orthogonal_adjust = (1.0 / (sqrt(numNode))) * sum
      //val orthognal_v = v.mapValues{(id,vi) => vi - (orthogonal_adjust)}
      //val v_orthonormal = normalizeV(orthognal_v);
      if (last_iter > 15) {
        println("Construct error graph")
      val error_graph = pgraph_v2.joinVertices(npgraph_v2.vertices) { case (id, v1, v2) => (v1 - v2) * (v1 - v2) };

        println("Compute error")
      //val error_rdd = nv.join(v_prime).mapValues( { case (v1, v2)=> })
      error = sqrt(error_graph.vertices.reduce { case ((id1, v1), (id2, v2)) => if (id1 > id2) {
        (id1, v1 + v2)
      } else {
        (id2, v1 + v2)
      }
      }._2)
      println("error = " + error)
      }
      pgraph_v1 = npgraph_v1;
      pgraph_v2 = npgraph_v2;


      last_iter = i
      //pgraph.vertices.foreach { case (id, attr) => println(id.toString + " : " + attr) }

    }

//    val sorted_vertecies = pgraph_v2.vertices.collect().sortBy(_._2)
//    for (j<-0 to numNode-2) {
//      gap.update(j, abs(sorted_vertecies.apply(j+1)._2 - sorted_vertecies.apply(j)._2));
//    }
//    val (gapVal, index) = gap.zipWithIndex.maxBy(_._1)
//
//    val gapEntry = sorted_vertecies.apply(index)._2
//    println("index= " + index + " gap entry =" + gapEntry);
//    val group1= pgraph_v2.subgraph(vpred= {case (id, value) => value >= gapEntry })
//    val group2 = pgraph_v2.subgraph(vpred= {case (id, value) => value < gapEntry })
//    //val numGroup1Edges = group1.edges.mapValues(e=>1).reduce{case (e1,e2) => Edge(e1.attr+e2.attr) };
//    //val numGroup1Edges = group1.edges.mapValues(e=>1).reduce{case (e1,e2) => Edge(e1.attr+e2.attr) };

    val v2_vector = pgraph_v2.vertices.map{case (id, f1)=> Vectors.dense(f1)}
    println("Running K-Means")
    val clusters = KMeans.train(v2_vector, 2, 100)

    val group1= pgraph_v2.subgraph(vpred= {case (id, value) => clusters.predict( Vectors.dense(value)) >= 1.0 })
    val group2= pgraph_v2.subgraph(vpred= {case (id, value) => clusters.predict( Vectors.dense(value)) < 1.0 })


    val totalEdges = pgraph_v2.edges.count();
    val groupAEdges = group1.edges.count();
    val groupBEdges = group2.edges.count();;
    //pa.partition{case (id, value)=> value <= gapEntry}
    val gAnodes = group1.vertices.count();
    val gBnodes = group2.vertices.count();
    println("A nodes = " + gAnodes + ", A edges = " + groupAEdges)
    println("B nodes = " + gBnodes + ", B edges = " + groupBEdges)
    //gap.foreach(g=>println(g))
    //val assoc_a_v = pgraph_v2.triplets.map{case t1=> if (t1.srcAttr >= gapEntry) {1.0} else {0.0}}.reduce((a,b)=>a+b)
    val assoc_a_v = pgraph_v2.triplets.map{case t1=> if (clusters.predict(Vectors.dense(t1.srcAttr)) >= 1.0) {1.0} else {0.0}}.reduce((a,b)=>a+b)
    val assoc_b_v = totalEdges - assoc_a_v
    val cutAB =  totalEdges - groupAEdges - groupBEdges;
    val ncutA = cutAB / assoc_a_v;
    val ncutB = cutAB / assoc_b_v;
    println("ncutA= " + ncutA);
    println("ncutB= " + ncutB);

    println("ncut = " + (ncutA + ncutB))
    //pgraph_v1.vertices.foreach { case (id, attr) => println(id.toString + " : " + attr) }
    //pgraph_v2.vertices.foreach { case (id, attr) => println(id.toString + " : " + attr) }

    //println("norm = " + normVertices(pgraph.vertices))
    //println("error = " + error)
    println("num of iteration = " + last_iter )
    println("norm v1 = "  + normGraph(pgraph_v1));
    //println("norm v2 = "  + normGraph(pgraph_v2));
    println("product = " + dotGraph(pgraph_v1, pgraph_v2));

    //val v2 = pgraph.joinVertices(degreeRDD){case (id, value, degree)=> value };
    //val nv = normalize(v2);
    //degreeRDD.foreach({case (id, attr)=> println(id + " : " + attr) })
    //pgraph.vertices.foreach({case (id, attr)=> println(id + " : " + attr  )})



  }

}
