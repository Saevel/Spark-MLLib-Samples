package prv.zielony.spark.mllib.clustering

import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by kamil on 2017-05-23.
  */
object MLLibClusteringApplication extends App with CentroidExtraction {

  val context = new SparkContext(new SparkConf().setAppName("SparkMLLibApplication").setMaster("local[*]"))

  val data: RDD[Vector] = context.parallelize(Seq(
    Vectors.dense(1.0, 1.0, 1.0, 0.0),
    Vectors.dense(0.0, 0.0, 1.0, 1.0),
    Vectors.dense(7.0, 7.0, 6.0, 6.0),
    Vectors.dense(6.0, 6.0, 5.0, 7.0)
  ))

  val centroids = data.centroids(2)

  println(s"Centroids: $centroids")

  println("Clustering Results: ")
  val clusterer = new KMeansModel(data.centroids(2).toArray)

  val results = clusterer.predict(data)

  data.zip(results).foreach(println)
  // TODO: Jaki przypadek zasymulować?

  // Pomysł1: Przewidywanie grupy wiekowej na podstawie zainteresowań?

  // Pomysł2: Coś medycznego?
}
