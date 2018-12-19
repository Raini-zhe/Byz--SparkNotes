数据：

20 21
50 51
50 52
50 53
50 54
60 51
60 53
60 52
60 56
60 57
70 58
60 61
70 54
70 55
70 56
70 57
70 58
10 55
80 67
90 43
30 44
50 67
50 87
40 77
20 11
10 55
20 84
70 45
90 55
91 44
78 44
76 32
88 23
91 34
56 11
33 23
24 11


import org.apache.spark.{SparkContext, SparkConf}

object SecondarySort {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName(" Secondary Sort ").setMaster("local")
    val sc = new SparkContext(conf)
    val file = sc.textFile("hdfs://worker02:9000/test/secsortdata")

    val rdd = file.map(line => line.split("\t")).
      map(x => (x(0),x(1))).groupByKey().
      sortByKey(true).map(x => (x._1,x._2.toList.sortWith(_>_)))

    val rdd2 = rdd.flatMap{
      x =>
      val len = x._2.length
      val array = new Array[(String,String)](len)
      for(i <- 0 until len) {
        array(i) = (x._1,x._2(i))
      }
      array  
    }

    sc.stop()
  }
}



