package utils

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
 * Created by sara on 9/10/15.
  * * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object CCData {
  def loadData(sc:SparkContext, atr:Int ,MName:String ,filepath:String) : Array[RDD[LabeledPoint]]  = {
    val data = sc.textFile(filepath);
    val dataArray = new Array[RDD[LabeledPoint]](atr);
    for (i <- 0 to atr - 1 ) {
      val parsedData = data.map { line =>
        val parts = line.split(',')
        val len = parts.length
        try {

           LabeledPoint(parts(i).toDouble, Vectors.dense(parts.slice(i+1 , len).map(_.toDouble)))

        } catch {
          case e: NumberFormatException => {
            null
          }
        }
      }
      parsedData.filter(a => a!= null)
      dataArray.update(i, parsedData);

    }
    dataArray
  }

}
