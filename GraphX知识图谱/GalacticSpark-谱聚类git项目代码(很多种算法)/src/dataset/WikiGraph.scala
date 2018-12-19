package dataset

import org.apache.spark.graphx.GraphLoader
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import utils.{FileHandler, FileLogger, GalaxyException, MetaFile}

/**
  * Created by sara on 11/23/15.
  * * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
  */
object WikiGraph {
   def main(args: Array[String]) {
     try {
     val path = args(0)
     val log = FileLogger.open(args(2))
     val sampleNum = 10
     val output = args(1)
     val uritype = args(3)
     //val reg = args(2)
     //val metaFiles = FileHandler.loadInput(filename)
     //println("metaFiles = " + metaFiles(0).uri)


     val conf = new SparkConf().setAppName("wikigraph") //"spark://dyna6-162.cs.uoregon.edu:7077")
     val sc = new SparkContext(conf)
     val sqlContext = new SQLContext(sc)


     val graph = GraphLoader.edgeListFile(sc, path)

     val numNodes = graph.numVertices.toInt;
     val numEdges = graph.numEdges.toInt;
     FileLogger.println("Num node=" + numNodes);
     FileLogger.println("Num edges=" + numEdges);




     val verFile = FileHandler.getFullName(sc, uritype, "vertices")
     val edgeFile = FileHandler.getFullName(sc, uritype, "edges")


     //graph.vertices.saveAsObjectFile(verFile)
     val vertexDF = sqlContext.createDataFrame(graph.vertices)
     vertexDF.write.format("com.databricks.spark.csv").save(verFile)

     graph.edges.saveAsObjectFile(edgeFile)

     val objects = new Array[MetaFile](2)
     val verObj = new MetaFile
     verObj.objName = "VertexRDD"
     verObj.uri = verFile
     verObj.uriType = uritype

     val edgeObj = new MetaFile
     edgeObj.objName = "EdgeRDD"
     edgeObj.uri = edgeFile
     edgeObj.uriType = uritype

     objects.update(0, verObj)
     objects.update(1, edgeObj)
     FileHandler.dumpOutput(args(1), objects)
     //val f = sc.textFile(metaFiles(0).uri);
     //f.cache();
     //val partedf= f.repartition(60);
     //partedf.cache();

     FileLogger.close();

     //shorts.foreach(s=>println(s._1 + " : " + s._2));
 //    val hashingTF = new HashingTF(5000)
 //    val tf = hashingTF.transform(documents)
 //    tf.cache();
 //    val idf = new IDF(minDocFreq = 5).fit(tf)
 //    val tfidf = idf.transform(tf)
 //    //tfidf.foreach(println(_));
 //    val numFeatures = tfidf.first().size
 //    val numPoint = tfidf.count();
 //    //val vecs = tfidf.takeSample(false,50)
 //    //vecs.foreach(v=>println(v.numNonzeros))
 //    println("numPoint = " + numPoint)
 //    println("numFeatures = " + numFeatures)
     //val v = tfidf.first()
     //println(v);
     //val mat: RowMatrix = new RowMatrix(tfidf);
     //tfidf.unpersist();



     //val pc = mat.computePrincipalComponents(100);
     //val projected: RowMatrix = mat.multiply(pc)
     //println(projected.rows.first())
     } catch {
       case e: Exception => println(GalaxyException.getString(e))
     }
   }

 }
