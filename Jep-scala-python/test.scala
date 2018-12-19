
// spark-shell --master spark://192.168.101.170:7077 --driver-memory 5g --executor-memory 6g --total-executor-cores 10 --driver-cores 3

import jep.{Jep, JepConfig, NDArray}
import org.apache.spark.sql.SparkSession

object JepAddExample extends App {
  val spark = SparkSession.
    builder().
    master("local[2]").
    appName("FPGrowthTest").
    config("spark.defalut.parallelism", 10).
    config("spark.sql.shuffle.partitions", 10).
    getOrCreate()
  val sc = spark.sparkContext
  val sqlContext = spark.sqlContext

  val all_data = spark.createDataFrame(List(("13269805111", "13911107818", "call_src", 2, 2, "8", 15),("18611347039", "15933075283", "call_src", 2, 2, "40", 14),("18610644843", "13126724326", "call_dst", 2, 2, "29", 16),
    ("13681044780", "13084315779", "call_dst", 2, 9, "6", 9),("18612181034", "085186303389", "call_src", 0, 2, "5", 11),("13681044780", "18801190296", "call_dst", 2, 0, "6", 12),("18612181034", "13146027742", "call_src", 2, 2, "5", 16))).toDF("phone1","phone2","call_event","phone1_type","phone2_type","callDuration","hours")

  val group_all_data = all_data.rdd.map(x => (x.getAs[String]("phone1"), Seq( x.getAs[String]("phone1"), x.getAs[String]("phone2"),x.getAs[String]("call_event"), x.getAs[Int]("phone1_type"),x.getAs[Int]("phone2_type"), x.getAs[String]("callDuration"), x.getAs[Int]("hours")))).
    reduceByKey((x,y) => x +: Seq(y))

  val jep = new Jep()
  jep.runScript("/home/raini/jepExample/fake_leader_v4.py")
  val a = 2
  val b = 3
  // There are multiple ways to evaluate. Let us demonstrate them:
  jep.eval(s"c = add($a, $b)")
  val ans = jep.getValue("c").asInstanceOf[Long]
  println(ans)
  val ans2 = jep.invoke("add", a.asInstanceOf[AnyRef], b.asInstanceOf[AnyRef]).asInstanceOf[Long]
  println(ans2)



  import jep.{Jep, JepConfig, NDArray}
  val all_data = spark.createDataFrame(List(("13269805111", "13911107818", "call_src", 2, 2, "8", 15),("18611347039", "15933075283", "call_src", 2, 2, "40", 14),("18610644843", "13126724326", "call_dst", 2, 2, "29", 16),
    ("13681044780", "13084315779", "call_dst", 2, 9, "6", 9),("1008612181034", "85186303389", "call_src", 0, 2, "5", 11),("13681044780", "18801190296", "call_dst", 2, 0, "6", 12),("18612181034", "13146027742", "call_src", 2, 2, "5", 16))).toDF("phone1","phone2","call_event","phone1_type","phone2_type","callDuration","hours")

  val group_all_data = all_data.rdd.map(x => (x.getAs[String]("phone1"), Seq( x.getAs[String]("phone1"), x.getAs[String]("phone2"), x.getAs[Int]("phone1_type"),x.getAs[Int]("phone2_type"), x.getAs[String]("callDuration"), x.getAs[Int]("hours")))).
    reduceByKey((x,y) => x +: Seq(y))


  val select_data1 = group_all_data.mapValues{ x=> //x.toString().replaceAll("List","").toList}.take(3)
    val jep = new Jep(new JepConfig().addSharedModules("numpy"))
    jep.runScript("/home/raini/jepExample/add.py")
    val a = 1
    val b = 2
    val aa = "abcddef"
    val cc = x.toString().replaceAll("List","").replaceAll("\\(","\\[").replaceAll("\\)","\\]")
    println("+++" + cc.getClass.getSimpleName + "===")
    jep.eval(s"c = add($cc)")
    val ans = jep.getValue("c")//.asInstanceOf[Array]
    jep.close()
    ans
  }
  select_data1.take(5)



  import jep.{Jep, JepConfig, NDArray}
  //jep.eval(s"import pandas as pd")
  val jep0 = new Jep(new JepConfig().addIncludePaths(".").addSharedModules("numpy"))
  jep0.eval("import numpy");
  val f = Array( 1.0f, 2.1f, 3.3f, 4.5f, 5.6f, 6.7f )
  val nd = new NDArray(f, 3, 2)
  jep.eval("print('========')")
  jep.set("x", nd)




 }



