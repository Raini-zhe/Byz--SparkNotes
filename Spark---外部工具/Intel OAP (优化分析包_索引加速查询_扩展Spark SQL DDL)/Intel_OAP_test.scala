package test

import org.apache.spark.sql.SparkSession

object Intel_OAP_test {

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
  @transient
  val sc = spark.sparkContext
  val sqlContext = spark.sqlContext
  import sqlContext.implicits._
  spark.sql("show databases").show



  /** 官网例子 */
  spark.sql(s"""CREATE TEMPORARY TABLE oap_test (a INT, b STRING)
               | USING oap
               | OPTIONS (path 'hdfs:///oap-data-dir/')""".stripMargin)
  val data = (1 to 300).map { i => (i, s"this is test $i") }.toDF().createOrReplaceTempView("t")
  spark.sql("insert overwrite table oap_test select * from t")
  spark.sql("create oindex index1 on oap_test (a)")
  spark.sql("show oindex from oap_test").show()
  spark.sql("SELECT * FROM oap_test WHERE a = 1").show()
  spark.sql("drop oindex index1 on oap_test")


  /** 创建索引和显示索引 */
  /*
   * OAP Extended DDL:
   * Create Index: "CREATE SINDEX index_name ON table_name (column_name) USING [BTREE, BITMAP]"
   * Drop Index: "DROP SINDEX index_name on table_name"
   * Show Index: "SHOW SINDEX FROM table_name"
   */

  //spark.sql("CREATE SINDEX IF NOT EXISTS oap_btree_index ON oap_index (userid) USING BTREE")
  //spark.sql("DROP SINDEX IF EXISTs oap_btree_index ON oap_index")
  spark.sql("SHOW SINDEX FROM oap_index").show()
  //spark.sql("CREATE SINDEX IF NOT EXISTS parquet_btree_index ON parquet_index (userid) USING BTREE")
  //spark.sql("DROP SINDEX if EXISTS parquet_btree_index ON parquet_index")
  spark.sql("SHOW SINDEX FROM parquet_index").show()


  /** 自己写的例子 */

  spark.sql("show databases").show




}
