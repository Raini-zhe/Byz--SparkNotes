package dataset

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import utils.MetaFile

/**
 * Created by sara on 11/5/15.
  * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object Url {

  def saveUrl(sc: SparkContext): Unit ={
    var whole : RDD[LabeledPoint] =MLUtils.loadLibSVMFile(sc, ("/Users/sara/url_svmlight/Day" + 0 + ".svm"))
    for (i <- 1 to 120) {
      val examples: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, ("/Users/sara/url_svmlight/Day" + i + ".svm"))
      whole = whole union(examples)
    }

    whole.repartition(4);
    whole.saveAsObjectFile("file:/Users/sara/galaxy/data/url")
  }
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("url")
    val sc = new SparkContext(conf)

    //val objects = new Array[MetaFile](120)

  }

}
