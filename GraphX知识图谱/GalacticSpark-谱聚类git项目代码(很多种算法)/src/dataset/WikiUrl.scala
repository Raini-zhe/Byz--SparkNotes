package dataset

import org.apache.spark.{SparkConf, SparkContext}
import utils.{FileHandler, FileLogger, GalaxyException, MetaFile}

/**
  * Created by sara on 11/23/15.
  * * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
  */
object WikiUrl {
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
     //val uritype = metaFiles(0).uriType


     val conf = new SparkConf().setAppName("url") //"spark://dyna6-162.cs.uoregon.edu:7077")
     val sc = new SparkContext(conf)
     val f = sc.textFile(path);
     //f.cache();
     //val partedf= f.repartition(60);
     //partedf.cache();
     val documents = f.map {
       case s =>
         val parts = s.split(",");
         (parts(0), parts(1))
     }
     //println("Num of total documents:" + documents.count());
     //println(documents.take(2).apply(1)._2)
     //val filtered = documents.filter{ case (uri, short_abstract)=> short_abstract.exists(s=>s.matches(reg))};
     val docFile = FileHandler.getFullName(sc, uritype, "wikiurl")
     documents.saveAsObjectFile(docFile);
     //filtered.cache();
     //FileLogger.println("Num of matched items: " + filtered.count() )

     FileLogger.println("Num of items: " + documents.count() )
     documents.takeSample(false,sampleNum).foreach{case (id, uri)=> FileLogger.println(id + " " + uri)}

     val objects = new Array[MetaFile](1)
     val obj = new MetaFile
     obj.objName = "WikiUrl"
     obj.uri = docFile
     obj.uriType = uritype;

     objects.update(0, obj);
     FileHandler.dumpOutput(output, objects)
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
