package prv.zielony.spark.ml.filtering

import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.recommendation.ALS.Rating
import org.apache.spark.sql.SparkSession

object SparkMLALSApplication extends App {

  val session = SparkSession.builder.master("local[*]").appName("SparkMLALSApplication").getOrCreate

  val data = session.createDataFrame(
    Seq(
      Rating(1, 1, 0.7f),
      Rating(1, 2, 0.3f),
      Rating(1, 3, 0.8f),
      Rating(2, 1, 0.2f),
      Rating(2, 2, 0.9f),
      Rating(3, 3, 0.25f),
      Rating(3, 1, 0.8f)
    )
  )

  val input = session.createDataFrame(Seq(
    (3, 2)
  )).toDF("user", "item")

  val als = new ALS()
    .setMaxIter(5)
    .setRegParam(0.01)
    .setImplicitPrefs(true)
    .setUserCol("user")
    .setItemCol("item")
    .setRatingCol("rating")

  als.fit(data).transform(input).show
}
