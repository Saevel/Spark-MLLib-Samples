package prv.zielony.spark.mllib.filtering

import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.sql.SparkSession

object MLLibALSApplication extends App {

  val session = SparkSession.builder().master("local[*]").appName("SparkALSApplication").getOrCreate
  import session.implicits._

  val data = session.sparkContext.parallelize(Seq(
    Rating(1, 1, 0.7f),
    Rating(1, 2, 0.3f),
    Rating(1, 3, 0.8f),
    Rating(2, 1, 0.2f),
    Rating(2, 2, 0.9f),
    Rating(3, 3, 0.25f),
    Rating(3, 1, 0.8f)
  ))

  val als = new ALS

  val prediction = als.run(data).predict(3, 2)

  println(s"ALS Prediction: $prediction")
}