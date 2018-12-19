
//官网例子：

import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions.col

val dfA = spark.createDataFrame(Seq(
  (0, Vectors.dense(1.0, 1.0)),
  (1, Vectors.dense(1.0, -1.0)),
  (2, Vectors.dense(-1.0, -1.0)),
  (3, Vectors.dense(-1.0, 1.0))
)).toDF("id", "features")

val dfB = spark.createDataFrame(Seq(
  (4, Vectors.dense(1.0, 0.0)),
  (5, Vectors.dense(-1.0, 0.0)),
  (6, Vectors.dense(0.0, 1.0)),
  (7, Vectors.dense(0.0, -1.0))
)).toDF("id", "features")

val key = Vectors.dense(1.0, 0.0)

val brp = new BucketedRandomProjectionLSH().setBucketLength(2.0).setNumHashTables(3).setInputCol("features").setOutputCol("hashes")

val model = brp.fit(dfA)

// Feature Transformation
println("The hashed dataset where hashed values are stored in the column 'hashes':")
model.transform(dfA).show()

// Compute the locality sensitive hashes for the input rows, then perform approximate
// similarity join.
// We could avoid computing hashes by passing in the already-transformed dataset, e.g.
// `model.approxSimilarityJoin(transformedA, transformedB, 1.5)`
println("Approximately joining dfA and dfB on Euclidean distance smaller than 1.5:")
model.approxSimilarityJoin(dfA, dfB, 1.5, "EuclideanDistance").select(col("datasetA.id").alias("idA"),
    col("datasetB.id").alias("idB"),
    col("EuclideanDistance")).show()


+---+---+-----------------+                                                     
|idA|idB|EuclideanDistance|
+---+---+-----------------+
|  1|  4|              1.0|
|  0|  6|              1.0|
|  1|  7|              1.0|
|  3|  5|              1.0|
|  0|  4|              1.0|
|  3|  6|              1.0|
|  2|  7|              1.0|
|  2|  5|              1.0|
+---+---+-----------------+

model.approxSimilarityJoin(dfA, dfB, 20, "EuclideanDistance").show()

+--------------------+--------------------+-----------------+
|            datasetA|            datasetB|EuclideanDistance|
+--------------------+--------------------+-----------------+ // 计算dfB里与dfA 最相似的3份数据， EuclideanDistance=20距离越大越不相似
|[1,[1.0,-1.0],Wra...|[4,[1.0,0.0],Wrap...|              1.0|
|[0,[1.0,1.0],Wrap...|[6,[0.0,1.0],Wrap...|              1.0|
|[2,[-1.0,-1.0],Wr...|[4,[1.0,0.0],Wrap...| 2.23606797749979| // MinHash for Jaccard Distance
|[0,[1.0,1.0],Wrap...|[5,[-1.0,0.0],Wra...| 2.23606797749979|
|[1,[1.0,-1.0],Wra...|[5,[-1.0,0.0],Wra...| 2.23606797749979|
|[1,[1.0,-1.0],Wra...|[7,[0.0,-1.0],Wra...|              1.0|
|[3,[-1.0,1.0],Wra...|[5,[-1.0,0.0],Wra...|              1.0|
|[0,[1.0,1.0],Wrap...|[4,[1.0,0.0],Wrap...|              1.0|
|[3,[-1.0,1.0],Wra...|[6,[0.0,1.0],Wrap...|              1.0|
|[3,[-1.0,1.0],Wra...|[4,[1.0,0.0],Wrap...| 2.23606797749979|
|[2,[-1.0,-1.0],Wr...|[7,[0.0,-1.0],Wra...|              1.0|
|[2,[-1.0,-1.0],Wr...|[5,[-1.0,0.0],Wra...|              1.0|
+--------------------+--------------------+-----------------+


model.approxSimilarityJoin(dfA, dfA, 200, "EuclideanDistance").show()
+--------------------+--------------------+-----------------+
|            datasetA|            datasetB|EuclideanDistance|
+--------------------+--------------------+-----------------+ // 该算法重点是距离的设定（应该可以参考聚类算法中 到类簇中心的距离）
|[2,[-1.0,-1.0],Wr...|[1,[1.0,-1.0],Wra...|              2.0|
|[0,[1.0,1.0],Wrap...|[0,[1.0,1.0],Wrap...|              0.0|
|[1,[1.0,-1.0],Wra...|[2,[-1.0,-1.0],Wr...|              2.0|
|[0,[1.0,1.0],Wrap...|[3,[-1.0,1.0],Wra...|              2.0|
|[3,[-1.0,1.0],Wra...|[3,[-1.0,1.0],Wra...|              0.0|
|[2,[-1.0,-1.0],Wr...|[2,[-1.0,-1.0],Wr...|              0.0|
|[1,[1.0,-1.0],Wra...|[1,[1.0,-1.0],Wra...|              0.0|
|[3,[-1.0,1.0],Wra...|[0,[1.0,1.0],Wrap...|              2.0|
+--------------------+--------------------+-----------------+




// Compute the locality sensitive hashes for the input rows, then perform approximate nearest
// neighbor search.
// We could avoid computing hashes by passing in the already-transformed dataset, e.g.
// `model.approxNearestNeighbors(transformedA, key, 2)`
println("Approximately searching dfA for 2 nearest neighbors of the key:")
model.approxNearestNeighbors(dfA, key, 2).show()

+---+----------+--------------------+-------+
| id|  features|              hashes|distCol|
+---+----------+--------------------+-------+ // 计算dfA里与key最相似的两个
|  1|[1.0,-1.0]|[[-1.0], [-1.0], ...|    1.0|
|  0| [1.0,1.0]|[[0.0], [0.0], [-...|    1.0|
+---+----------+--------------------+-------+
































#
