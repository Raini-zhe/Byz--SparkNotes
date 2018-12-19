package graph

/**
 * Created by sara on 10/21/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */



import org.apache.spark.graphx._
import scala.math._


class NCut {
  def evaluate(g: Graph[Int, Int]) : Double = {
    var ncut = 0.0
    val numCluster = g.vertices.reduce{case ((id1,v1), (id2,v2))=> (-1, max(v1, v2))}._2 + 1
    for (i<-0 to numCluster - 1) {


      // i is the current cluster
      // and g is a graph in which
      // the attribute of each vertex shows its cluster
      val r = g.aggregateMessages[(Double,Double)](
          sendMsg = t => t.sendToSrc( if (t.srcAttr == i)
            { if (t.dstAttr == i) (0.0, 1.0) else (1.0, 1.0)}
              else (0.0, 0.0) ),
          mergeMsg = (m1,m2) =>(m1._1+m2._1, m1._2+m2._2))
      val (cut, assoc) = r.reduce({case ((id1,v1), (id2,v2))
                        => (-1, (v1._1+v2._1, v1._2 + v2._2))})._2


      ncut = ncut + cut/assoc
    }
    ncut
  }


}
