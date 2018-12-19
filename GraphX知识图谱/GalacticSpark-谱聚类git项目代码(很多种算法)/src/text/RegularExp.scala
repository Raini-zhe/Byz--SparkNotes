package text

import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by sara on 11/2/15.
  * * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object RegularExp {

  def main (args: Array[String]): Unit = {
    val file = args(0)
    val output = args(1)
    val reg = args(2)
    val conf = new SparkConf().setAppName("match") //"spark://dyna6-162.cs.uoregon.edu:7077")
    val sc = new SparkContext(conf)
    val f = sc.textFile(file);
    //sc.parallelize(f, 10);
    val matched = f.filter(s=> s.matches(reg))


  }

}
