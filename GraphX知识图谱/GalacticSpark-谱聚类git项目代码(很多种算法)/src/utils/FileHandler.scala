package utils

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.io.Source._
import java.io.{File, PrintWriter, FileInputStream}

/**
 * Created by sara on 11/3/15.
  * * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object FileHandler {

  def getFullName(sc: SparkContext, uritype: String, name: String): String = {
    val galaxy_home =  sys.env("GALAXY_HOME")
    return "file:" + galaxy_home + "tmp/" + name + "-" + sc.applicationId;
  }

//  def dumpRDD(meta: String, sc: SparkContext, uritype: String, name: String, rdd: RDD[LabeledPoint]): Unit = {
//    val fullname = getFullName(sc, uritype, name)
//    rdd.saveAsObjectFile(fullname)
//    val metaFile = new MetaFile
//    metaFile.uri = fullname
//    metaFile.objName = "RDD"
//    metaFile.uriType = uritype
//    dumpOutput(meta, Array(metaFile))
//
//  }

  def loadInput(meta : String) : Array[MetaFile] = {
    val lines = scala.io.Source.fromFile(meta).getLines()
    var num = 0
    if (lines.hasNext) {
      num = lines.next().toInt
    }
    var objects = new Array[MetaFile](num)
    var i = 0;
    while (lines.hasNext ) {
      val l = lines.next();
      //println(l)
      val objUri = l.split(' ')
      //objUri.foreach(s=>println(s))
      val objName = objUri.apply(0)
      val uriType = objUri.apply(1)
      val uri = objUri.apply(2)
      val metaFile = new MetaFile
      metaFile.objName = objName;
      metaFile.uri = uri;
      metaFile.uriType = uriType
      objects.update(i, metaFile)
      i = i + 1;
    }
    return objects
  }


  def dumpOutput(meta: String, objects: Array[MetaFile]) : Unit = {
    val pw1 = new PrintWriter(new File(meta));
    pw1.println(objects.length.toString)
    objects.foreach( m=>pw1.println(m.objName + " " + m.uriType + " " + m.uri))
    pw1.close()
  }
}
