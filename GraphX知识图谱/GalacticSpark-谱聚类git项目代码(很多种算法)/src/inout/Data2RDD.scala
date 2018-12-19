package inout

import org.apache.spark.graphx.GraphLoader
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import utils._
/**
 * Created by sara on 11/6/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object Data2RDD {
  def main(args: Array[String]) {

    val filename = args(0)
    val output= args(1)
    val log = args(2)
    val inputType = args(3)
    val uriType = args(4)
    val classLabel = args(5)

    FileLogger.open(log)
    val hasLabel = args(6).toBoolean
    val hasHeader = args(7).toBoolean
    val outputType = args(8)


    try {


      val conf = new SparkConf().setAppName("data2rdd")
      val sc = new SparkContext(conf)
      val sqlC = new SQLContext(sc)

      val metaFiles = FileHandler.loadInput(filename)
      val uritype = metaFiles(0).uriType
      val file = metaFiles(0).uri

      if (inputType.equals("libsvm")) {
        val outputRDD: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, file)
        if (outputType == "LabeledPointRDD") {
          val fullname = FileHandler.getFullName(sc, uriType, "svmdata")
          outputRDD.saveAsObjectFile(fullname)
          val metaFile = new MetaFile("RDD", uriType, fullname)
          FileHandler.dumpOutput(output, Array(metaFile))
          FileLogger.println("Number of points:" + outputRDD.count())
        }else{

          val outputFile = FileHandler.getFullName(sc, uritype, "lable_point_df")
          val df = sqlC.createDataFrame(outputRDD).toDF("label", "features")
          FileLogger.println("Scheme string: " + "label, features")
          df.write
            .format("com.databricks.spark.csv")
            .option("header", "true")
            .save(outputFile)

          //ToCSV.fromVertexRDD(sqlC, degree, degreeFile)

          //val objects = new Array[MetaFile](1)
          val metafile = new MetaFile("CSV[lable,features]", uriType, outputFile )
          FileHandler.dumpOutput(output, Array(metafile))
//          verObj.objName = "CSV[vertex,rank]"
//          verObj.uri = rankFile
//          verObj.uriType = uritype

          //objects.update(0, verObj)
          //FileHandler.dumpOutput(output, objects)

        }
      }

      if (inputType.equals("csv")) {
        val dataDF = Data.loadCSV(sqlC, file, hasHeader, false)


        if (hasLabel) {
          val labelIndex = try {
            classLabel.toInt
          } catch {
            case e: Exception => dataDF.columns.indexOf(classLabel)
          }

          val outputRDD: RDD[LabeledPoint] = dataDF.rdd.map { r =>
            val f = r.toSeq.toArray.zipWithIndex.filter { case (c, i) => i != labelIndex }
              .map { case (c, i) => c.asInstanceOf[String].toDouble }
            LabeledPoint(r.getString(labelIndex).toDouble, Vectors.dense(f))
          }
          FileLogger.println("Feature Size:" + outputRDD.take(10).apply(2).features.size)
          if (outputType == "LabeledPointRDD") {
            val fullname = FileHandler.getFullName(sc, uriType, "csvdata")
            outputRDD.saveAsObjectFile(fullname)
            val metaFile = new MetaFile("RDD[LabeledPoint]", uriType, fullname)
            FileHandler.dumpOutput(output, Array(metaFile))
            FileLogger.println("Number of points:" + outputRDD.count())
          }else{
            /////////////////////////////
            val outputFile = FileHandler.getFullName(sc, uritype, "lable_point_df")
            val df = sqlC.createDataFrame(outputRDD).toDF("label", "features")
            FileLogger.println("Scheme string: " + "label, features")
            df.write
              .format("com.databricks.spark.csv")
              .option("header", "true")
              .save(outputFile)

            //ToCSV.fromVertexRDD(sqlC, degree, degreeFile)

            //val objects = new Array[MetaFile](1)
            val metafile = new MetaFile("CSV[lable,features]", uriType, outputFile )
            FileHandler.dumpOutput(output, Array(metafile))
          }

        } else {
          //For unlabeled points the label is zero

          val outputRDD: RDD[LabeledPoint] = dataDF.rdd.map { r =>
            val f = r.toSeq.toArray.map { case c => c.asInstanceOf[String].toDouble }
            LabeledPoint(0.0, Vectors.dense(f))
          }
          if (outputType == "LabeledPointRDD") {
            val fullname = FileHandler.getFullName(sc, uriType, "csvdata")
            outputRDD.saveAsObjectFile(fullname)
            val metaFile = new MetaFile("RDD[LabeledPoint]", uriType, fullname)
            FileHandler.dumpOutput(output, Array(metaFile))
            FileLogger.println("Number of points:" + outputRDD.count())

          }else{
            //////////////////////////
            val outputFile = FileHandler.getFullName(sc, uritype, "lable_point_df")
            val df = sqlC.createDataFrame(outputRDD).toDF("label", "features")
            FileLogger.println("Scheme string: " + "label, features")
            df.write
              .format("com.databricks.spark.csv")
              .option("header", "true")
              .save(outputFile)

            //ToCSV.fromVertexRDD(sqlC, degree, degreeFile)

            //val objects = new Array[MetaFile](1)
            val metafile = new MetaFile("CSV[lable,features]", uriType, outputFile )
            FileHandler.dumpOutput(output, Array(metafile))
          }
        }
      }

      /*if (inputType.equals("libsvm") || inputType.equals("csv") || inputType.equals("tsv")) {
      if (hasLabel) {
        val outputRDD: RDD[LabeledPoint] = if (inputType.equals("libsvm")) {
          MLUtils.loadLibSVMFile(sc, file)
        } else if (inputType.equals("tsv")) {
          Data.loadTSV2Labeled(sc, file)
        } else {
          Data.loadData(sc, file)
        }
        //outputRDD.saveAsObjectFile(FileHandler.getFullName(sc, "local", "data"))

        val fullname = FileHandler.getFullName(sc, "LOCAL", "data")
        outputRDD.saveAsObjectFile(fullname)
        val metaFile = new MetaFile("RDD", "LOCAL", fullname)
        FileHandler.dumpOutput(outfile, Array(metaFile))

        //FileHandler.dumpRDD(outfile, sc, "local", "data", outputRDD)
      } else {
        val outputRDD: RDD[Vector] = if (inputType.equals("tsv")) {
          Data.loadTSV2Vector(sc, file)
        } else {
          Data.loadData2Vector(sc, file)
        }
        //outputRDD.saveAsObjectFile(FileHandler.getFullName(sc, "local", "data"))

        val fullname = FileHandler.getFullName(sc, "LOCAL", "data")
        outputRDD.saveAsObjectFile(fullname)
        val metaFile = new MetaFile("RDD", "LOCAL", fullname)
        FileHandler.dumpOutput(outfile, Array(metaFile))

      }

    } */

//      if (inputType.equals("edgeview")) {
//        val uritype = uriType
//        val graph = GraphLoader.edgeListFile(sc, file)
//        val outVerFile = FileHandler.getFullName(sc, uritype, "vertices")
//        val outEdgeFile = FileHandler.getFullName(sc, uritype, "edges")
//        graph.vertices.saveAsObjectFile(outVerFile)
//        graph.edges.saveAsObjectFile(outEdgeFile)
//        val verObj = new MetaFile("VertexRDD", uritype, outVerFile)
//        val edgeObj = new MetaFile("EdgeRDD", uritype, outEdgeFile)
//        FileHandler.dumpOutput(output, Array(verObj, edgeObj))
//        FileLogger.println("Number of verteces: " + graph.vertices.count())
//        FileLogger.println("Number of edges: " + graph.edges.count())
//      }
      FileLogger.println("Data loaded sucessfully.")

    } catch {
      case e: Exception => FileLogger.println(GalaxyException.getString(e))

      //case e: Exception => FileLogger.println("ERROR. tool unsuccessful:" + e);
    } finally {
      FileLogger.close()
    }
  }

}
