package graph


import java.io.{FileOutputStream, File, PrintWriter}

import org.apache.spark.graphx._
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import utils.{GalaxyException, MetaFile, FileHandler, FileLogger}
import Array._
import org.apache.spark.mllib.linalg._

import scala.collection.mutable.HashMap

/**
  * Created by sara on 6/24/15.
  * * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
  */
object BiSpectral {






  def main(args: Array[String]) {
    val filename = args(0);
    val output = args(1);
    val log = args(2)
    val output2 = args(3)
    val numIteration = args(4).toInt
    FileLogger.open(log);

    try {
      val start = System.currentTimeMillis()

      val conf = new SparkConf().setAppName("bispectral");
      val sc = new SparkContext(conf)
      val sqlC = new SQLContext(sc);


      val metaFiles = FileHandler.loadInput(filename)
      val vFile = metaFiles(0).uri
      val eFile = metaFiles(1).uri
      val vertexRDD = sc.objectFile[(VertexId, Any )](vFile)
      val edgeRDD = sc.objectFile[org.apache.spark.graphx.Edge[Any]](eFile);
      val uritype =metaFiles(0).uriType
      val inputgraph = Graph(vertexRDD, edgeRDD)
      val numNode = inputgraph.numVertices.toInt;
      FileLogger.println("Num node=" + numNode);


      val fspectral = new FSpectral()
      val clusterGraph = fspectral.run(inputgraph, 2, numIteration )


      val sub1 = clusterGraph.subgraph(vpred = (id, attr) => attr == 0 )
      val sub2 = clusterGraph.subgraph(vpred = (id, attr) => attr == 1 )

      /*val clusterFile = FileHandler.getFullName(sc, uritype, "cluster")
      */

      val verFile1 = FileHandler.getFullName(sc, uritype, "vertices")
      val edgeFile1 = FileHandler.getFullName(sc, uritype, "edges")

      sub1.vertices.saveAsObjectFile(verFile1)
      sub1.edges.saveAsObjectFile(edgeFile1)

      val verObj1 = new MetaFile("VertexRDD[Int]", uritype, verFile1)
      val edgeObj1 = new MetaFile("EdgeRDD[Int]", uritype, edgeFile1)

      FileHandler.dumpOutput(output, Array(verObj1, edgeObj1))

      val verFile2 = FileHandler.getFullName(sc, uritype, "vertices")
      val edgeFile2 = FileHandler.getFullName(sc, uritype, "edges")

      sub2.vertices.saveAsObjectFile(verFile2)
      sub2.edges.saveAsObjectFile(edgeFile2)

      val verObj2 = new MetaFile("VertexRDD[Int]", uritype, verFile2)
      val edgeObj2 = new MetaFile("EdgeRDD[Int]", uritype, edgeFile2)
      FileHandler.dumpOutput(output2, Array(verObj2, edgeObj2))

      val stop = System.currentTimeMillis()

      FileLogger.println("BiSpectral successffuly done in " + ((stop - start)/1000.0) + " sec.")


      /*
      val schemaString = "vertex,cluster"
      val df = sqlC.createDataFrame(clusterGraph.vertices).toDF("vertex", "cluster")
      FileLogger.println("Scheme string: " + schemaString)
      df.write
        .format("com.databricks.spark.csv")
        .option("header", "true")
        .save(clusterFile)
      df.take(10).foreach(FileLogger.println(_))
      //ToCSV.fromVertexRDD(sqlC, degree, degreeFile)

      val verObj = new MetaFile
      verObj.objName = "CSV[vertex,cluster]"
      verObj.uri = clusterFile
      verObj.uriType = uritype

      FileHandler.dumpOutput(output, Array(verObj))
*/
      /*
            clusterGraph.vertices.saveAsObjectFile(verFile)
            clusterGraph.edges.saveAsObjectFile(edgeFile)
            objects.update(0, verObj)
            objects.update(1, edgeObj)
            FileHandler.dumpOutput(args(1), objects)
        */
    } catch {
      case e: Exception => FileLogger.println(GalaxyException.getString(e))
    } finally {
      FileLogger.close();
    }

  }


  def matrixToRDD(sc:SparkContext, m: Matrix): RDD[Vector] = {
    val columns = m.toArray.grouped(m.numRows)
    val rows = columns.toSeq.transpose // Skip this if you want a column-major RDD.
    val vectors = rows.map(row => new DenseVector(row.toArray))
    sc.parallelize(vectors)
  }


  def oldmain (args: Array[String]) = {
    //val filename = "file:/Users/sara/drp/adjList.txt"
    val filename = args(0);
    val conf = new SparkConf().setAppName("bispectral").setMaster("spark://cn168:7077"); //"spark://dyna6-162.cs.uoregon.edu:7077")
    val sc = new SparkContext(conf)
    val inputgraph = GraphLoader.edgeListFile(sc,filename)
    val numNode = inputgraph.numVertices.toInt;
    val mapping = new HashMap[Long,Int]();
    val revMapping = new HashMap [Int,Long]();

    //inputgraph.vertices.for

    val verticeIds = inputgraph.vertices.map( u => u._1 ).collect();
    for(i<-0 to numNode - 1) {
      mapping.put(verticeIds.apply(i), i)
      revMapping.put(i,verticeIds.apply(i));
    }

    val nVertices = inputgraph.vertices.map( u=> (mapping.apply(u._1).toLong, u._2))
    val nEdges = inputgraph.edges.map(e=> Edge(mapping.apply(e.srcId).toLong, mapping.apply(e.dstId).toLong, e.attr))
    val ngraph = Graph(nVertices, nEdges);
    /*inputgragh.edges.collect().foreach(println(_))
    inputgragh.vertices.collect().foreach(println(_))*/
    //val pw = new PrintWriter(new File(args(1)));
    // val pw = new PrintWriter(new File("file:" + args(1)));


    val output = ngraph.collectNeighborIds(EdgeDirection.Out)
    /*output.collect().foreach(println)*/
    //for (x <- output){
    //  print("\n")
    //  print(x._1 + " ")
    //  print("[")
    //  for (y <- x._2)
    //    print(y + ",")
    //  print("]")
    //  print("\n")
    //}
    /*val vlen = output.mapValues(r => r.length)
    val vnormal = output.mapValues(r => r.map(e=>(1.0/(r.length)) ))
    val vlenout = output.leftJoin(vlen)((x,y,z) => (z,y))
    val vlenoutnorm = vlenout.leftJoin(vnormal)((x, y, z) => ( y, z)) */
   // val numNode = inputgraph.numVertices.toInt;
    val spvec = output.mapValues(r => (Vectors.sparse( numNode,  r.map(e=>e.toInt) , r.map(e=>(1.0/(r.length)) ))))
    //val spvec = output.map( r => (Vectors.sparse( numNode, r._2.map(e=>e.toInt) : r.map)))
    val deg = output.mapValues(r => r.length );
    val degrow = deg.map(v=>v._2.toDouble);
    val rows = spvec.map(v=>v._2); //  .map(((vid, v)) => v) // map[(VertexId, Vector)]( v=>v);
    val order = spvec.map(v=>v._1);
    //println(order.collect().foreach(println(_)))
    val mat = new RowMatrix(rows);

    //val numberofEigenvectores = 2
    val numberofEigenvectores = args(3).toInt
    val pc = mat.computePrincipalComponents(numberofEigenvectores)
    //val dm: Matrix = Matrices.dense(5, 5, Array(0, 0.5, 0.0, 0, 0,  1.0, 0, 0.333,0, 0,    0, 0.5, 0, 0.5,0.5,   0,0,0.333, 0, 0.5,   0,0,0.333, 0.5,0));


    val pcRDD = matrixToRDD(sc, pc);
    //val numberOfClusters = 2
    val numberOfClusters = 2;
    val clusters = KMeans.train(pcRDD, numberOfClusters, 100)

    val clusterArray = pcRDD.map(p=> clusters.predict(p) ).collect();
    val assignedClusters = order.map( o => (o, clusterArray.apply(o.toInt))).collect();

    val sub0 = ngraph.subgraph(vpred = (id, attr) => {clusterArray.apply(id.toInt) == 0});
    val sub1 = ngraph.subgraph(vpred = (id, attr) => clusterArray.apply(id.toInt) == 1);


    val pw1 = new PrintWriter(new File(args(1)));
    val pw2 = new PrintWriter(new File(args(2)));
    sub0.edges.collect().foreach(e=>{pw1.write(revMapping.apply(e.srcId.toInt) + " " + revMapping(e.dstId.toInt) + "\n")})
    sub1.edges.collect().foreach(e=>{pw2.write(revMapping.apply(e.srcId.toInt) + " " + revMapping(e.dstId.toInt)+ "\n")});
    //sub0.edges.saveAsTextFile(args(1));
    //sub1.edges.saveAsTextFile(args(2));
    pw1.close();
    pw2.close();
      //assignedClusters.foreach(node=> {pw.write(node._1.toString); pw.write(" ");pw.write(node._2.toString);pw.write("\n")})
      //pw.close();
      //    val degvec = (Vectors.dense(degrow.collect()))
      //  for( a <- 0 to (numNode - 1)) {
      //  val b = pc.apply(a, 0) * degvec.apply(order.collect().apply(a).toInt)
      //  println(a, b, degvec.apply(a));
      //}
      //println(pc.toString());

      //println(spvec.collect());

      /* val fact = inputgragh.triplets.collect().foreach(println(_))
      inputgragh.aggregateMessages( edgeTriplet: EdgeTriplet  => Iterator((edge.srcId,1),(edge.dstId,1)).collect()*/

    }



}
