package prv.zielony.spark.ml.clustering

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}

object MLClusteringApplication extends App with CentroidExtraction {

  val session = SparkSession.builder().master("local[*]").appName("SparkMLApplication").getOrCreate()

  val data = session.createDataFrame(
    Seq(
      Patient(1, 12, 0, true, true),
      Patient(2, 24, 1, true, false),
      Patient(3, 24, 0, true, true),
      Patient(4, 20, 0, false, true),
      Patient(5, 27, 1, false, false)
    )
  )

  val vectorIndexer = new VectorAssembler().setInputCols(Array("age", "race", "sex")).setOutputCol("features")

  var model = new KMeans().setK(3).setMaxIter(30).setFeaturesCol("features").setPredictionCol("group")
  //(data.centroids("features", 3)).setFeaturesCol("features").setPredictionCol("group")

  val pipeline = new Pipeline().setStages(Array(vectorIndexer, model))

  val result: DataFrame = pipeline.fit(data).transform(data)

  result.show

  case class Patient(id: Long, age: Int, race: Int, sex: Boolean, therapySuccessful: Boolean)
}
