package ml


//import org.apache.spark.mllib.clustering.{LDA, DistributedLDAModel}

import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by sara on 9/11/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object LDATopic {
  def main(args: Array[String]) {
    try {

      val conf = new SparkConf().setAppName("lda").setMaster("local")
      val sc = new SparkContext(conf)

      // Load and parse the data
      val data = sc.textFile("file:/Users/sara/drp/spark/spark-1.4.1/data/mllib/sample_lda_data.txt")
      val parsedData = data.map(s => Vectors.dense(s.trim.split(' ').map(_.toDouble)))
      // Index documents with unique IDs
      val corpus = parsedData.zipWithIndex.map(_.swap).cache()

      // Cluster the documents into three topics using LDA
      val ldaModel = new LDA().setK(3).run(corpus)

      // Output topics. Each is a distribution over words (matching word count vectors)
      println("Learned topics (as distributions over vocab of " + ldaModel.vocabSize + " words):")
      val topics = ldaModel.topicsMatrix
      for (topic <- Range(0, 3)) {
        print("Topic " + topic + ":")
        for (word <- Range(0, ldaModel.vocabSize)) { print(" " + topics(word, topic)); }
        println()
      }

      // Save and load model. version 1.5
      //ldaModel.save(sc, "myLDAModel")
      //val sameModel = DistributedLDAModel.load(sc, "myLDAModel")

    }catch {
      case e: Exception => println("ERROR. tool unsuccessful: " + e);
    }
  }

}
