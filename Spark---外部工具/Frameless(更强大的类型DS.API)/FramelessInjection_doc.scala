package test

import org.apache.spark.sql.SparkSession
import frameless.TypedDataset
import frameless._

object FramelessInjection_doc {
  val spark = SparkSession.
    builder().
    master("spark://192.168.101.170:7077").
    appName("GraphPhoneFraudTest").
    config("spark.sql.shuffle.partitions", 300)
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.executor.memory", "30g")
    .config("spark.driver.memory", "25g")
    .config("spark.network.timeout", 300)
    .config("spark.default.parallelism", 200)
    .config("spark.driver.maxResultSize", "6g")
    .config("spark.shuffle.consolidateFilesx", "true").
    config("spark.sql.warehouse.dir", "/home/EverSecAnalysis/warehouse/").
    enableHiveSupport().
    getOrCreate()
  @transient val sc = spark.sparkContext
  @transient val sqlContext = spark.sqlContext
  import sqlContext.implicits._

  case class Person(age: Int, birthday: java.util.Date)
  val people = Seq(Person(42, new java.util.Date))

  val aptDs = spark.createDataset(people)

  implicit val dateToLongInjection = new Injection[java.util.Date, Long] {
    def apply(d: java.util.Date): Long = d.getTime()
    def invert(l: Long): java.util.Date = new java.util.Date(l)
  }
  // 等价于： implicit val dateToLongInjection = Injection((_: java.util.Date).getTime(), new java.util.Date((_: Long)))

  val personDS = TypedDataset.create(people )
  // personDS: frameless.TypedDataset[Person] = [age: int, birthday: bigint]

}
