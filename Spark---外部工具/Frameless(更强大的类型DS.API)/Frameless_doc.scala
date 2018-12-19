package test

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import frameless.functions.aggregate//._
import frameless.TypedDataset
import frameless._

import frameless.syntax._  // .show.run()


object Frameless_doc {

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



  /** 创建TypedDataset实例 */
  case class Apartment(city: String, surface: Int, price: Double, bedrooms: Int)
  val apartments = Seq(
    Apartment("Paris", 50,  300000.0, 2),
    Apartment("Paris", 100, 450000.0, 3),
    Apartment("Paris", 25,  250000.0, 1),
    Apartment("Lyon",  83,  200000.0, 2),
    Apartment("Lyon",  45,  133000.0, 1),
    Apartment("Nice",  74,  325000.0, 3)
  )
  /* 1. 该方法报错
    * <console>:41: error: could not find implicit value for parameter sqlContext: org.apache.spark.sql.SparkSession
    */
  val aptTypedDs1 = TypedDataset.create(apartments)

  /* 2. 该方法可以 */
  val aptDs = spark.createDataset(apartments)
  val aptTypedDs = TypedDataset.create(aptDs)
  aptTypedDs.show().run()

  /* 3. 该方法可以 */
  frameless.Injection
  val data = (1 to 300).map { i => (i, s"this is test $i") }.toDS
  val aptTypedDs3 = data.typed
  aptTypedDs3.show().run()

  /** 类型安全的列引用 */
  val cities: TypedDataset[String] = aptTypedDs.select(aptTypedDs('city))

  // select() 支持任意列操作：
  aptTypedDs.select(aptTypedDs('surface) * 10, aptTypedDs('surface) + 2).show().run()

  // 提取子集
  case class UpdatedSurface(city: String, surface: Int)
  // defined class UpdatedSurface
  val updated = aptTypedDs.select(aptTypedDs('city), aptTypedDs('surface) + 2).as[UpdatedSurface]
  updated.show().run()
  //  +-----+-------+
  //  | city|surface|
  //  +-----+-------+
  //  |Paris|     52|
  //  |Paris|    102|
  //  |Paris|     27|
  //  | Lyon|     85|
  //  | Lyon|     47|
  //  | Nice|     76|
  //  +-----+-------+

  //
  val aptds = aptTypedDs // For shorter expressions
  // aptds: frameless.TypedDataset[Apartment] = [city: string, surface: int ... 2 more fields]
  case class ApartmentDetails(city: String, price: Double, surface: Int, ratio: Double)
  // defined class ApartmentDetails
  val aptWithRatio =
    aptds.select(
      aptds('city),
      aptds('price),
      aptds('surface),
      aptds('price) / aptds('surface).cast[Double]  // <---- 类型转换
    ).as[ApartmentDetails]

  // 提取子集 (.project)
  case class ApartmentShortInfo(city: String, price: Double, bedrooms: Int)
  val aptTypedDs2: TypedDataset[ApartmentShortInfo] = aptTypedDs.project[ApartmentShortInfo]
  aptTypedDs2.show().run
  //  +-----+--------+--------+
  //  | city|   price|bedrooms|
  //  +-----+--------+--------+
  //  |Paris|300000.0|       2|
  //  |Paris|450000.0|       3|
  //  |Paris|250000.0|       1|
  //  | Lyon|200000.0|       2|
  //  | Lyon|133000.0|       1|
  //  | Nice|325000.0|       3|
  //  +-----+--------+--------+


  /** TypedDataset函数和转换
    *
    */
  import frameless.functions._                // For literals
  import frameless.functions.nonAggregate._   // e.g., concat, abs
  import frameless.functions.aggregate._      // e.g., count, sum, avg
  // dropTupled() 删除单个列并生成基于元组的模式。

  aptTypedDs2.dropTupled('price): TypedDataset[(String,Int)]
  // : frameless.TypedDataset[(String, Int)] = [_1: string, _2: int]

  // 通常，您想用新值替换现有的列。
  val inflation = aptTypedDs2.withColumnReplaced('price, aptTypedDs2('price) * 2)
  inflation.show().run
  //  +-----+--------+--------+
  //  | city|   price|bedrooms|
  //  +-----+--------+--------+
  //  |Paris|600000.0|       2|
  //  |Paris|900000.0|       3|
  //  |Paris|500000.0|       1|
  //  | Lyon|400000.0|       2|
  //  | Lyon|266000.0|       1|
  //  | Nice|650000.0|       3|
  //  +-----+--------+--------+

  /*** ---------------------------------------- 删除/替换/添加字段 ------------- */
  //dropTupled() 删除单个列并生成基于元组的模式--说是列表更合适。

  aptTypedDs2.dropTupled('price): TypedDataset[(String,Int)]
  // res17: frameless.TypedDataset[(String, Int)] = [_1: string, _2: int]

  //删除列并指定新的模式使用drop() -- （也就是删除无关列 == select
  case class CityBeds(city: String, bedrooms: Int)
  val cityBeds: TypedDataset[CityBeds] = aptTypedDs2.drop[CityBeds]
  // cityBeds: frameless.TypedDataset[CityBeds] = [city: string, bedrooms: int]


  //通常，您想用新值替换现有的列。
  val inflation = aptTypedDs2.withColumnReplaced('price, aptTypedDs2('price) * 2)
  // inflation: frameless.TypedDataset[ApartmentShortInfo] = [city: string, price: double ... 1 more field]
  inflation.show(2).run()
  // +-----+--------+--------+
  // | city|   price|bedrooms|
  // +-----+--------+--------+
  // |Paris|600000.0|       2|
  // |Paris|900000.0|       3|
  // +-----+--------+--------+
  // only showing top 2 rows
  //
  val cityBeds: TypedDataset[CityBeds] = aptTypedDs2.drop[CityBeds]

  // 或者用固定值代替。
  import frameless.functions.lit
  val res2 = aptTypedDs2.withColumnReplaced('price, lit(0.001))
  res2.show().run
  //  +-----+-----+--------+
  //  | city|price|bedrooms|
  //  +-----+-----+--------+
  //  |Paris|0.001|       2|
  //  |Paris|0.001|       3|
  //  |Paris|0.001|       1|
  //  | Lyon|0.001|       2|
  //  | Lyon|0.001|       1|
  //  | Nice|0.001|       3|
  //  +-----+-----+--------+


  // 使用withColumnTupled()基于tupled的模式中的结果添加列。
  aptTypedDs2.withColumnTupled(lit(Array("a","b","c"))).show(2).run()
  //  +-----+--------+---+---------+
  //  |   _1|      _2| _3|       _4|
  //  +-----+--------+---+---------+ < --- (增加了一列在最后)
  //  |Paris|300000.0|  2|[a, b, c]|
  //  |Paris|450000.0|  3|[a, b, c]|
  //  +-----+--------+---+---------+

  // 增加列，并对结果赋予Schema
  case class CityBedsOther(city: String, price: Double, bedrooms: Int, other: List[String])
  // defined class CityBedsOther
  aptTypedDs2.
    withColumn[CityBedsOther](lit(List("a","b","c"))). // CityBedsOther字段名要与父类一致，增加的列名可以自定义
    show(1).run()
  //  +-----+--------+--------+---------+
  //  | city|   price|bedrooms|    other|
  //  +-----+--------+--------+---------+
  //  |Paris|300000.0|       2|[a, b, c]|
  //  +-----+--------+--------+---------+


  //要有条件地更改列，请使用该when/otherwise操作。
  import frameless.functions.nonAggregate.when
  aptTypedDs2.withColumnTupled(
    when(aptTypedDs2('city) === "Paris" && aptTypedDs2('bedrooms) === 2, aptTypedDs2('price)).
      when(aptTypedDs2('city) === "Lyon", lit(1.1)).
      otherwise(lit(0.0))).show(8).run()
  //  +-----+--------+---+--------+
  //  |   _1|      _2| _3|      _4|
  //  +-----+--------+---+--------+
  //  |Paris|300000.0|  2|300000.0|
  //  |Paris|450000.0|  3|450000.0|
  //  |Paris|250000.0|  1|250000.0|
  //  | Lyon|200000.0|  2|     1.1|
  //  | Lyon|133000.0|  1|     1.1|
  //  | Nice|325000.0|  3|     0.0|
  //  +-----+--------+---+--------+
  // 要有条件地更改列，when/otherwise 条件并列。
  aptTypedDs2.withColumnTupled(
    when(aptTypedDs2('city) === "Paris" && aptTypedDs2('bedrooms) === 2, aptTypedDs2('price)).
      when(aptTypedDs2('city) === "Lyon", lit(1.1)).
      otherwise(lit(0.0))).show(8).run()
  //  +-----+--------+---+--------+
  //  |   _1|      _2| _3|      _4|
  //  +-----+--------+---+--------+
  //  |Paris|300000.0|  2|300000.0|
  //  |Paris|450000.0|  3|     0.0|
  //  |Paris|250000.0|  1|     0.0|
  //  | Lyon|200000.0|  2|     1.1|
  //  | Lyon|133000.0|  1|     1.1|
  //  | Nice|325000.0|  3|     0.0|
  //  +-----+--------+---+--------+


  // 在不丢失重要模式信息的情况下添加列的简单方法是使用该asCol()方法将整个源模式投影到单个列中。返回嵌套列
  // == df.select($"*", ...)
  val c = aptTypedDs2.select(aptTypedDs2.asCol, lit(List("a","b","c"))) // lit并增加一列
  c.show(2).run()
  //  +------------------+---------+
  //  |                _1|       _2|
  //  +------------------+---------+
  //  |[Paris,300000.0,2]|[a, b, c]|
  //  |[Paris,450000.0,3]|[a, b, c]|
  //  +------------------+---------+

  // 要访问嵌套列，请使用 colMany()方法。
  c.select(c.colMany('_1, 'city), c('_2)).show(2).run()
  //  +-----+---------+
  //  |   _1|       _2|
  //  +-----+---------+
  //  |Paris|[a, b, c]|
  //  |Paris|[a, b, c]|
  //  +-----+---------+


  /** ---------------------------------------------使用集合 -------------------------- */
  import frameless.functions._
  import frameless.functions.nonAggregate._

  val t = aptTypedDs2.select(aptTypedDs2('city), lit(List("abc","c","d")))
  // t: frameless.TypedDataset[(String, List[String])] = [_1: string, _2: array<string>]
  t.withColumnTupled(
    arrayContains(t('_2), "abc")
  ).show(3).run()
  //  +-----+-----------+----+
  //  |   _1|         _2|  _3|
  //  +-----+-----------+----+
  //  |Paris|[abc, c, d]|true|
  //  |Paris|[abc, c, d]|true|
  //  |Paris|[abc, c, d]|true|
  //  +-----+-----------+----+

  // || 匹配集合多个元素
  t.withColumnTupled(
    arrayContains(t('_2), "abc") || arrayContains(t('_2), "r")
  ).show(3).run()
  //  +-----+-----------+----+
  //  |   _1|         _2|  _3|
  //  +-----+-----------+----+
  //  |Paris|[abc, c, d]|true|
  //  |Paris|[abc, c, d]|true|
  //  |Paris|[abc, c, d]|true|
  //  +-----+-----------+----+

  // && 匹配集合多个元素
  t.withColumnTupled(
    arrayContains(t('_2), "abc") && arrayContains(t('_2), "r")
  ).show(3).run()
  //  +-----+-----------+-----+
  //  |   _1|         _2|   _3|
  //  +-----+-----------+-----+
  //  |Paris|[abc, c, d]|false|
  //  |Paris|[abc, c, d]|false|
  //  |Paris|[abc, c, d]|false|
  //  +-----+-----------+-----+

  /** --------------------------------------------- 将数据收集到驱动程序 --------------------------
    * // 在Frameless中，所有Spark动作（如collect()）都是安全的，需要run.
    * */

  // 从数据集中获取第一个元素（如果数据集为空返回None）。
  cityBeds.headOption.run()
  // res27: Option[CityBeds] = Some(CityBeds(Paris,2)) <---------(返回Schema

  //采取第一个n要素。
  cityBeds.take(2).run()
  // res28: Seq[CityBeds] = WrappedArray(CityBeds(Paris,2), CityBeds(Paris,3))
  cityBeds.head(3).run()
  // res29: Seq[CityBeds] = WrappedArray(CityBeds(Paris,2), CityBeds(Paris,3), CityBeds(Paris,1))
  cityBeds.limit(4).collect().run()
  // res30: Seq[CityBeds] = WrappedArray(CityBeds(Paris,2), CityBeds(Paris,3), CityBeds(Paris,1), CityBeds(Lyon,


  /** --------------------------------------------- 对列进行排序 --------------------------
    * 只能选择可排序的列类型进行排序。
    * .asc .desc
    * */
  aptTypedDs.orderBy(aptTypedDs('city).asc).show(9).run()
  //  +-----+-------+--------+--------+
  //  | city|surface|   price|bedrooms|
  //  +-----+-------+--------+--------+
  //  | Lyon|     83|200000.0|       2|
  //  | Lyon|     45|133000.0|       1|
  //  | Nice|     74|325000.0|       3|
  //  |Paris|    100|450000.0|       3|
  //  |Paris|     50|300000.0|       2|
  //  |Paris|     25|250000.0|       1|
  //  +-----+-------+--------+--------+
  aptTypedDs.orderBy(
    aptTypedDs('city).asc,
    aptTypedDs('price).desc
  ).show(12).run()
  //  +-----+-------+--------+--------+
  //  | city|surface|   price|bedrooms|
  //  +-----+-------+--------+--------+
  //  | Lyon|     83|200000.0|       2|
  //  | Lyon|     45|133000.0|       1|
  //  | Nice|     74|325000.0|       3|
  //  |Paris|    100|450000.0|       3|
  //  |Paris|     50|300000.0|       2|
  //  |Paris|     25|250000.0|       1|
  //  +-----+-------+--------+--------+


  /** --------------------------------------------- 用户定义的功能 udf --------------------------
    *
    * 无框架支持将任何Scala函数（最多五个参数）提升到特定的上下文TypedDataset：
    * */

  // The function we want to use as UDF
  val priceModifier =
    (name: String, price:Double) => if(name == "Paris") price * 2.0 else price
  // priceModifier: (String, Double) => Double = <function2>

  val udf = aptTypedDs.makeUDF(priceModifier)
  // udf: (frameless.TypedColumn[Apartment,String], frameless.TypedColumn[Apartment,Double]) => frameless.TypedColumn[Apartment,Double] = <function2>

  val aptds = aptTypedDs // For shorter expressions
  // aptds: frameless.TypedDataset[Apartment] = [city: string, surface: int ... 2 more fields]

  val adjustedPrice = aptds.select(aptds('city), udf(aptds('city), aptds('price)))
  // adjustedPrice: frameless.TypedDataset[(String, Double)] = [_1: string, _2: double]

  adjustedPrice.show().run()
  // +-----+--------+
  // |   _1|      _2|
  // +-----+--------+
  // |Paris|600000.0|
  // |Paris|900000.0|
  // |Paris|500000.0|
  // | Lyon|200000.0|
  // | Lyon|133000.0|
  // | Nice|325000.0|
  // +-----+--------+
  //


  /** --------------------------------------------- GroupBy和Aggregations --------------------------
    *
    *
    * */

  // 假设我们想要检索每个城市的平均公寓价格
  val priceByCity = aptTypedDs.groupBy(aptTypedDs('city)).agg(avg(aptTypedDs('price)))
  // priceByCity: frameless.TypedDataset[(String, Double)] = [_1: string, _2: double]
  priceByCity.collect().run()
  // Seq[(String, Double)] = WrappedArray((Nice,325000.0), (Paris,333333.3333333333), (Lyon,166500.0))

  // 接下来，我们结合select并groupBy计算每个城市的平均价格/面积比率：
  val aptds = aptTypedDs // For shorter expressions
  // aptds: frameless.TypedDataset[Apartment] = [city: string, surface: int ... 2 more fields]
  val cityPriceRatio =  aptds.select(aptds('city), aptds('price) / aptds('surface).cast[Double])
  // cityPriceRatio: frameless.TypedDataset[(String, Double)] = [_1: string, _2: double]
  cityPriceRatio.groupBy(cityPriceRatio('_1)).agg(avg(cityPriceRatio('_2))).show().run()
  // +-----+------------------+
  // |   _1|                _2|
  // +-----+------------------+
  // | Nice| 4391.891891891892|
  // |Paris| 6833.333333333333|
  // | Lyon|2682.5970548862115|
  // +-----+------------------+
  //


  /** 我们还可以用 pivot 来在辅助列上进一步分组数据。例如，我们可以通过卧室数量来比较城市间的平均价格。
    * 语法：
    *     DS .groupBy(aptds('city))  .pivot(aptds('bedrooms))  .on(1,2,3,4)    达到细分groupBy的目的
    *
    * */

  case class BedroomStats(
                           city: String,
                           AvgPriceBeds1: Option[Double], // Pivot values may be missing, so we encode them using Options
                           AvgPriceBeds2: Option[Double],
                           AvgPriceBeds3: Option[Double],
                           AvgPriceBeds4: Option[Double])
  val bedroomStats = aptds.
    groupBy(aptds('city)).
    pivot(aptds('bedrooms)).
    on(1,2,3,4).           //  根据4个房间分组 算平均
    agg(avg(aptds('price))).
    as[BedroomStats]  // Typesafe casting

  bedroomStats.show().run()
  // +-----+-------------+-------------+-------------+-------------+
  // | city|AvgPriceBeds1|AvgPriceBeds2|AvgPriceBeds3|AvgPriceBeds4|
  // +-----+-------------+-------------+-------------+-------------+
  // | Nice|         null|         null|     325000.0|         null|
  // |Paris|     250000.0|     300000.0|     450000.0|         null|
  // | Lyon|     133000.0|     200000.0|         null|         null|
  // +-----+-------------+-------------+-------------+-------------+
  //  scala> aptds.show().run()
  //  +-----+-------+--------+--------+
  //  | city|surface|   price|bedrooms|
  //  +-----+-------+--------+--------+
  //  |Paris|     50|300000.0|       2|
  //  |Paris|    100|450000.0|       3|
  //  |Paris|     25|250000.0|       1|
  //  | Lyon|     83|200000.0|       2|
  //  | Lyon|     45|133000.0|       1|
  //  | Nice|     74|325000.0|       3|
  //  +-----+-------+--------+--------+

  /** 挑选使用可选字段 - null值处理
    *
    * 使用可选字段可以转换为非可选字段getOrElse()。
    * */

  val sampleStats = bedroomStats.select(
    bedroomStats('AvgPriceBeds2).getOrElse(0.0), // 类型要一致
    bedroomStats('AvgPriceBeds3).getOrElse(11.0))
  // sampleStats: frameless.TypedDataset[(Double, Double)] = [_1: double, _2: double]
  sampleStats.show().run()
  //  +--------+--------+
  //  |      _1|      _2|
  //  +--------+--------+
  //  |300000.0|450000.0|
  //  |200000.0|    11.0|
  //  |     0.0|325000.0|
  //  +--------+--------+


  /** 整个TypedDataset聚合
    *
    *   我们经常想要聚合整个TypedDataset并跳过这个groupBy()子句。
    *   在无框架中，您可以agg()直接使用操作员进行此操作TypedDataset。
    *   在以下示例中，我们计算整个数据集的平均价格，平均曲面，最小曲面以及城市集合。
    *
    *   .agg 参数最多5个
    * */
  case class Stats(
                    avgPrice: Double,
                    avgSurface: Double,
                    minSurface: Int,
                    allCities: Vector[String])
  aptds.agg(
    avg(aptds('price)),
    avg(aptds('surface)),
    min(aptds('surface)),
    collectSet(aptds('city)) //,
    //aggregate.corr(aptds('surface), aptds('surface))
  ).as[Stats].show().run()
  // +-----------------+------------------+----------+-------------------+
  // |         avgPrice|        avgSurface|minSurface|          allCities|
  // +-----------------+------------------+----------+-------------------+
  // |276333.3333333333|62.833333333333336|        25|[Paris, Nice, Lyon]|
  // +-----------------+------------------+----------+-------------------+


  // 您也可以将任何TypedColumn操作应用于TypedAggregate列。
  aptds.agg(
    avg(aptds('price)) * min(aptds('surface)).cast[Double], // 比如两列做 乘法
    avg(aptds('surface)) * 0.2,
    litAggr("Hello World")
  ).show().run()
  // +-----------------+------------------+-----------+
  // |               _1|                _2|         _3|
  // +-----------------+------------------+-----------+
  // |6908333.333333333|12.566666666666668|Hello World|
  // +-----------------+------------------+-----------+


  /** ----------------------------------------------------------- （ Join ）---------------------------------
    *
    * 与DF不同： 两边可以有相同名字的列
    *
    * .colMany : 选择struct类型里的元素
    *
    *  */

  case class CityPopulationInfo(name: String, population: Int, surface: Int)
  val cityInfo = Seq(
    CityPopulationInfo("Paris", 2229621, 3),
    CityPopulationInfo("Lyon", 500715, 3),
    CityPopulationInfo("Nice", 343629, 3)
  )
  val cityInfoDs = spark.createDataset(cityInfo)
  val citiInfoTypedDS = TypedDataset.create(cityInfoDs)

  // 以下是如何将人口信息加入公寓的数据集：
  val withCityInfo = aptTypedDs.joinInner(citiInfoTypedDS) { aptTypedDs('city) === citiInfoTypedDS('name) }
  // withCityInfo: frameless.TypedDataset[(Apartment, CityPopulationInfo)] = [_1: struct<city: string, surface: int ... 2 more fields>, _2: struct<name: string, population: int>]
  withCityInfo.toDF()

  withCityInfo.show(9,false).run()
  //  +----------------------+---------------+
  //  |_1                    |_2             |
  //  +----------------------+---------------+
  //  |[Paris,25,250000.0,1] |[Paris,2229621]|
  //  |[Paris,50,300000.0,2] |[Paris,2229621]|
  //  |[Paris,100,450000.0,3]|[Paris,2229621]|
  //  |[Lyon,83,200000.0,2]  |[Lyon,500715]  |
  //  |[Lyon,45,133000.0,1]  |[Lyon,500715]  |
  //  |[Nice,74,325000.0,3]  |[Nice,343629]  |
  //  +----------------------+---------------+


  // 加入的TypedDataset具有类型TypedDataset[(Apartment, CityPopulationInfo)]。
  // 然后，我们可以选择我们想要继续使用的信息：

  case class AptPriceCity(city: String, aptPrice: Double, cityPopulation: Int)

  withCityInfo.select(
    withCityInfo.colMany('_2, 'name), withCityInfo.colMany('_1, 'price), withCityInfo.colMany('_2, 'population)
  ).as[AptPriceCity].show().run
  // +-----+--------+--------------+
  // | city|aptPrice|cityPopulation|
  // +-----+--------+--------------+
  // |Paris|300000.0|       2229621|
  // |Paris|450000.0|       2229621|
  // |Paris|250000.0|       2229621|
  // | Lyon|200000.0|        500715|
  // | Lyon|133000.0|        500715|
  // | Nice|325000.0|        343629|
  // +-----+--------+--------------+
  //

}
