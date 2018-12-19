package graph


import org.apache.spark.sql.SQLContext

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.graphx._
import utils.{FileLogger, GalaxyException, MetaFile, FileHandler}

/**
 * Created by sara on 10/13/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object GraphClustering {


  def main(args: Array[String]): Unit = {
    val filename = args(0)
    val output = args(1)
    val log = args(2)
    val output2 = args(3)

    val alg = args(4)
    val clusterNumStr = args(5)
    val numIterationStr = args(6)
    val eigsNumStr = args(7)

    FileLogger.open(log)

    try {
      val start = System.currentTimeMillis()
      val clusterNum = clusterNumStr.toInt
      val eigsNum = eigsNumStr.toInt
      val numIteration = numIterationStr.toInt

      val conf = new SparkConf().setAppName("clustering")
      conf.set("spark.scheduler.mode", "FAIR")
      val sc = new SparkContext(conf)
      val sqlC = new SQLContext(sc)

      val metaFiles = FileHandler.loadInput(filename)
      val vFile = metaFiles(0).uri
      val eFile = metaFiles(1).uri
      val vertexRDD = sc.objectFile[(VertexId, Any )](vFile)
      val edgeRDD = sc.objectFile[org.apache.spark.graphx.Edge[Any]](eFile)

      val graph = Graph(vertexRDD, edgeRDD)
      //val inputgraph = GraphLoader.edgeListFile(sc, metaFiles(0).uri)
      val uritype = metaFiles(0).uriType

      val numNode = graph.numVertices.toInt
      FileLogger.println("Num node=" + numNode)


      val clusterGraph = if (alg == "pca") {
        if (numNode >= 65536) {
          FileLogger.println("PCAClustering does not work for graphs with more than 65536 nodes.")
          throw GalaxyException
        }

        FileLogger.println("Running PCA clustering.")
        val pcaClustering = new PCAClustering()
        pcaClustering.run(graph, clusterNum, eigsNum, sc)
      } else if (alg == "spectral") {

        FileLogger.println("Warning: Ignoring given number of eigenvectors.")
        FileLogger.println("Running Spectral clustering using two eigenvectors.")

        val fspectral = new FSpectral()
        fspectral.run(graph, clusterNum, numIteration )
      } else if (alg == "random") {

        FileLogger.println("Running random clustering.")
        val random = new RandomClustering()
        random.run(graph, clusterNum)
      } else { // (alg == "PIC") {

        FileLogger.println("Running PIC clustering.")
        val picClustering = new PICClustering
        picClustering.run(graph, clusterNum, numIteration)
      }



      //val clusterGraph = fspectral.run(inputgraph, numIteration )

      val clusterFile = FileHandler.getFullName(sc, uritype, "cluster")

      //if (outputType == "csv") {

      val schemaString = "vertex,cluster"
      val df = sqlC.createDataFrame(clusterGraph.vertices).toDF("vertex", "cluster")
      FileLogger.println("Scheme string: " + schemaString)
      df.write
        .format("com.databricks.spark.csv")
        .option("header", "true")
        .save(clusterFile)
      df.take(10).foreach(FileLogger.println(_))
      //ToCSV.fromVertexRDD(sqlC, degree, degreeFile)

      val csvObj = new MetaFile
      csvObj.objName = "CSV[vertex,cluster]"
      csvObj.uri = clusterFile
      csvObj.uriType = uritype

      FileHandler.dumpOutput(output, Array(csvObj))



      val verFile = FileHandler.getFullName(sc, uritype, "vertices")
      val edgeFile = FileHandler.getFullName(sc, uritype, "edges")

      clusterGraph.vertices.saveAsObjectFile(verFile)
      clusterGraph.edges.saveAsObjectFile(edgeFile)



      val verObj = new MetaFile("VertexRDD[Int]", uritype, verFile)
      val edgeObj = new MetaFile("EdgeRDD[Int]", uritype, edgeFile)
      FileHandler.dumpOutput(output2, Array(verObj, edgeObj))
      //FileLogger.println(output2)

      val stop = System.currentTimeMillis()

      FileLogger.println("Clustering successffuly done in " + ((stop - start)/1000.0) + " sec.")

    } catch {
      case e: Exception => FileLogger.println(GalaxyException.getString(e))
    } finally {
      FileLogger.close()
    }
    //val modularity = Modularity.evaluate(clusterGraph)

    //val ncut = NCut.evaluate(clusterGraph)

    //val pw2 = new PrintWriter(new File(args(1)));
    //pw2.println("modularity = " + modularity);
    //pw2.println("ncut = " + ncut)
    //pw2.close();
  }

}
