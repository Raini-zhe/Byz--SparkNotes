package text

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.sql.SQLContext
import utils.{Data, FileHandler, GalaxyException, FileLogger}
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}

/**
  * Created by sara on 1/21/16.
  ** Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
  */
object Word2VecTool {

  def main(args: Array[String]) {
    try {
      val filename = args(0)
      val log = args(2)
      val output = args(1)
      val conf = new SparkConf().setAppName("word2vec")
      val sc = new SparkContext(conf)
      val sqlC = new SQLContext(sc)

      val metaFiles = FileHandler.loadInput(filename)
      val uritype = metaFiles(0).uriType
      val uri = metaFiles(0).uri

      FileLogger.open(log);
      val df =  Data.loadCSV(sqlC, uri,false, false)
      val documents = df.map(r=>r.getAs[String](0).split(" ").toSeq)
      val word2vec = new Word2Vec()
      val model = word2vec.fit(documents);
      model.findSynonyms("mathematic", 20).foreach{case (s, c)=> FileLogger.println(s + " : " + c.toString)}
      model.save(sc, output)

    } catch {
      case e: Exception => println(GalaxyException.getString(e))
    }
  }

}
