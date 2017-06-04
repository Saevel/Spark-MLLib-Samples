package prv.zielony.spark.mllib.clustering

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

import scala.util.Try

trait CentroidExtraction {

  implicit class CentroidFactory(vectors: RDD[Vector]) {

    def centroids(n: Int): List[Vector] = chooseCentroids(vectors)(n)
  }

  private def chooseCentroids(vectors: RDD[Vector])(n: Int): List[Vector] = {
    var centroids = vectors.take(1).headOption.toList

    (0 to n - 2).foreach{ _ =>
      centroids.lastOption
        .flatMap{ current =>
          nextCentroid(vectors, current).toOption
        }
        .foreach{ centroid => centroids = centroids :+ centroid}
    }

    centroids
  }

  private def nextCentroid(vectors: RDD[Vector], current: Vector): Try[Vector] = Try {
    vectors.map(vector =>
      (vector, Vectors.sqdist(vector, current))
    ).reduce((first, second) => if (first._2 >= second._2) first else second)._1
  }
}
