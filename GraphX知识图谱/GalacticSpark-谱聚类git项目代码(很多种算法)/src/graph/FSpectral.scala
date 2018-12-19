package graph

import org.apache.spark.graphx.{TripletFields, PartitionStrategy, Graph}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import utils.FileLogger

import scala.math._
import scala.util.Random

/**
  * Created by sara on 1/25/16.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
  */
class FSpectral {

  def gram_schmidt(g: Graph[(Double,Double), Double]): Graph[(Double,Double), Double ] = {

    //val v1_dot_v2 = g.vertices.map{ case (id,(f1,f2))=> f1*f2}.reduce((f1,f2)=>f1 + f2);
    //val v1_norm = g.vertices.map{ case (id,(f1,f2))=> f1*f1}.reduce((f1,f2)=>f1 + f2);

    val v1_dot_v2_plus_v1_norm = g.vertices.map{
      case (id,(f1,f2))=> (f1*f2, f1*f1)
    }.reduce{ case (f1,f2)=>(f1._1 + f2._1, f1._2 + f2._2) }

    val v1_dot_v2 = v1_dot_v2_plus_v1_norm._1
    val v1_norm = v1_dot_v2_plus_v1_norm._2

    val coef = v1_dot_v2 / v1_norm; //dotValue / normValue;
    //val orthogonal_g = g.mapVertices{ case (id, (v1, v2)) => (v1, v2 - (v1 * coef ) )}
    //val v2_norm = orthogonal_g.vertices.map{ case (id,(f1,f2))=> f2*f2}.reduce((f1,f2)=>f1 + f2);
    //val orthonormal_g = orthogonal_g.mapVertices{ case (id, (f1,f2)) => (f1/v1_norm, f2/v2_norm)}

    val orthogonal_g = g.mapVertices{ case (id, (v1, v2)) => (v1, pow(v2 - (v1 * coef ), 2) )}
    val v2_norm = orthogonal_g.vertices.reduce{
      case ( (id1,(f1,f2)), (id2, (v1,v2)) ) =>
        if (id1> id2) (id1,(f1,f2 + v2)) else (id2, (f1,f2 + v2))
    }._2._2

    //        reduce{case (f1,f2)=>f1 + f2};

    val orthonormal_g = orthogonal_g.mapVertices{ case (id, (f1,f2)) => (f1/v1_norm, sqrt(f2)/v2_norm)}
    orthonormal_g
  }


  def gram_schmidt_simple(g: Graph[(Double,Double), Double]): Graph[(Double,Double), Double ] = {
    val v1_dot_v2 = g.vertices.map{ case (id,(f1,f2))=> f1*f2}.reduce((f1,f2)=>f1 + f2)
    val v1_norm = g.vertices.map{ case (id,(f1,f2))=> f1*f1}.reduce((f1,f2)=>f1 + f2)
    val coef = v1_dot_v2 / v1_norm
    val orthogonal_g = g.mapVertices{ case (id, (v1, v2)) => (v1, pow(v2 - (v1 * coef ), 2) )}
    val v2_norm = orthogonal_g.vertices.reduce{ case ( (id1,(f1,f2)), (id2, (v1,v2)) )
    =>  if (id1> id2) (id1,(f1,f2 + v2)) else (id2, (f1,f2 + v2))}._2._2
    val orthonormal_g = orthogonal_g.mapVertices{ case (id, (f1,f2)) => (f1/v1_norm, sqrt(f2)/v2_norm)}
    orthonormal_g
  }

  def run(inputGraph: Graph[Any, Any], clusterNum: Int, iterationNum: Int): Graph[Int, Any] = {

    Random.setSeed(System.currentTimeMillis())

    val degreeRDD = inputGraph.outDegrees

    val agraph = Graph(degreeRDD, inputGraph.edges)

    val pgraph = agraph.mapTriplets { triplet => 1.0 / (sqrt(triplet.srcAttr) * sqrt(triplet.dstAttr)) }

    //FileLogger.println("Creating (v1,v2):")
    val v1v2 = pgraph.vertices.map { case (vertexId, a) => (vertexId, (Random.nextDouble(), Random.nextDouble())) }
    var pgraph_v1v2 = Graph(v1v2, pgraph.edges).partitionBy(PartitionStrategy.RandomVertexCut)

    pgraph_v1v2.cache()
    inputGraph.unpersist()
    var error = 10.0
    var last_iter = 0

    //FileLogger.println("Run gram-schmidth for random vectors")
    val opgraph_v1v2 = gram_schmidt(pgraph_v1v2)
    pgraph_v1v2 = opgraph_v1v2
    pgraph_v1v2.cache()
    for (i <- 0 to iterationNum if error > 0.0001) {
      //FileLogger.println(" =============== iteration: " + i + " ====================")
      //FileLogger.println("Compute P*(v1,v2)");

      val v1v2_vertices = pgraph_v1v2.aggregateMessages[(Double, Double)](

        sendMsg = ctx => ctx.sendToSrc((ctx.attr * ctx.dstAttr._1, ctx.attr * ctx.dstAttr._2)),

        mergeMsg = (msg1, msg2) => (msg1._1 + msg2._1, msg1._2 + msg2._2),

        TripletFields.Dst)



      val orthonormal_g = gram_schmidt(Graph(v1v2_vertices, pgraph_v1v2.edges))


      if (last_iter > iterationNum) { // never happens (so we never compute expensive error value)
        FileLogger.println("Construct error graph")
        val error_graph = pgraph_v1v2.joinVertices(orthonormal_g.vertices) {
          case (id, (v1, v2), (f1, f2)) => (pow(v1 - f1,2), pow(v2 - f2,2)) }
        FileLogger.println("Compute error")
        val error_v1v2 = error_graph.vertices.reduce {
          case ((id1, (v1, v2)), (id2, (f1, f2))) =>
            if (id1 > id2) {(id1, (v1 + f1, v2 + f2))
            } else { (id2, (v1 + f1, v2 + f2))}
        }._2

        error = error_v1v2._2
        FileLogger.println("error = " + error)
      }


      pgraph_v1v2 = orthonormal_g
      pgraph_v1v2.cache()

      last_iter = i

    }


    val v2_vector = pgraph_v1v2.vertices.map { case (id, (f1, f2)) => Vectors.dense(f2) }.cache()
    //FileLogger.println("Running K-Means")
    val clusters = KMeans.train(v2_vector, clusterNum, 100)
    val clustersV = pgraph_v1v2.mapVertices { case (id, (v1, v2)) => clusters.predict(Vectors.dense(v2))}

    Graph(clustersV.vertices, inputGraph.edges)


  }

}
