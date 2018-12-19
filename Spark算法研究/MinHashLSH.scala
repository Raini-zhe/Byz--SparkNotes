
import org.apache.spark.ml.feature.MinHashLSH
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions.col

val dfA = spark.createDataFrame(Seq(
  (0, Vectors.sparse(6, Seq((0, 1.0), (1, 1.0), (2, 1.0)))),
  (1, Vectors.sparse(6, Seq((2, 1.0), (3, 1.0), (4, 1.0)))),
  (2, Vectors.sparse(6, Seq((0, 1.0), (2, 1.0), (4, 1.0))))
)).toDF("id", "features")

val dfB = spark.createDataFrame(Seq(
  (3, Vectors.sparse(6, Seq((1, 1.0), (3, 1.0), (5, 1.0)))),
  (4, Vectors.sparse(6, Seq((2, 1.0), (3, 1.0), (5, 1.0)))),
  (5, Vectors.sparse(6, Seq((1, 1.0), (2, 1.0), (4, 1.0))))
)).toDF("id", "features")

val key = Vectors.sparse(6, Seq((1, 1.0), (3, 1.0)))

val mh = new MinHashLSH().setNumHashTables(5).setInputCol("features").setOutputCol("hashes")

val model = mh.fit(dfA)

// Feature Transformation
println("The hashed dataset where hashed values are stored in the column 'hashes':")
model.transform(dfA).show()

// Compute the locality sensitive hashes for the input rows, then perform approximate
// similarity join.
// We could avoid computing hashes by passing in the already-transformed dataset, e.g.
// `model.approxSimilarityJoin(transformedA, transformedB, 0.6)`
println("Approximately joining dfA and dfB on Jaccard distance smaller than 0.6:")
model.approxSimilarityJoin(dfA, dfB, 0.6, "JaccardDistance").select(col("datasetA.id").alias("idA"),
    col("datasetB.id").alias("idB"),
    col("JaccardDistance")).show()

// Compute the locality sensitive hashes for the input rows, then perform approximate nearest
// neighbor search.
// We could avoid computing hashes by passing in the already-transformed dataset, e.g.
// `model.approxNearestNeighbors(transformedA, key, 2)`
// It may return less than 2 rows when not enough approximate near-neighbor candidates are
// found.
println("Approximately searching dfA for 2 nearest neighbors of the key:")
model.approxNearestNeighbors(dfA, key, 2).show(false)
+---+-------------------------+--------------------------------------------------------------------------------------+-------+
|id |features                 |hashes                                                                                |distCol|
+---+-------------------------+--------------------------------------------------------------------------------------+-------+
|0  |(6,[0,1,2],[1.0,1.0,1.0])|[[-2.031299587E9], [-1.974869772E9], [-1.974047307E9], [4.95314097E8], [7.01119548E8]]|0.75   |
+---+-------------------------+--------------------------------------------------------------------------------------+-------+







def approxNearestNeighbors(dataset: Dataset[_], key: Vector, numNearestNeighbors: Int, distCol: String): Dataset[_]
  Given a large dataset and an item, approximately find at most k items which have the closest distance to the item. If the outputCol is missing, the method will transform the data; if the outputCol exists, it will use the outputCol. This allows caching of the transformed data when necessary.
dataset
  The dataset to search for nearest neighbors of the key.
key
  Feature vector representing the item to search for.
numNearestNeighbors
  The maximum number of nearest neighbors.
distCol
  Output column for storing the distance between each result row and the key.
returns
  A dataset containing at most k items closest to the key. A column "distCol" is added to show the distance between each row and the key.

This method is experimental and will likely change behavior in the next release.




#
