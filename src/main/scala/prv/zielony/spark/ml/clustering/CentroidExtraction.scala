package prv.zielony.spark.ml.clustering

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, Encoder, Row}

import scala.util.Try

trait CentroidExtraction {

  protected implicit class DataFrameWithCentroids(data: DataFrame) {

    def centroids(dataColumn: String, n: Int)(implicit e1: Encoder[Vector], e2: Encoder[(Row, Double)]) =
      chooseCentroids(data, n, dataColumn)
  }

  private def chooseCentroids(data: DataFrame, n: Int, dataColumn: String)(implicit e1: Encoder[Vector], e2: Encoder[(Row, Double)]): List[Vector] = {
    var centroids = data.map(_.getAs[Vector](dataColumn)).take(1).headOption.toList

    (0 to n - 2).foreach{ _ =>
      centroids.lastOption
        .flatMap{ current =>
          nextCentroid(data, dataColumn, current).toOption
        }
        .foreach{ centroid => centroids = centroids :+ centroid}
    }

    centroids
  }

  private def nextCentroid(data: DataFrame, dataColumn: String, current: Vector)
                          (implicit e1: Encoder[(Row, Double)], e2: Encoder[Vector]): Try[Vector] = Try {
    data.map(vector =>
      (vector, Vectors.sqdist(vector.getAs[Vector](dataColumn), current))
    ).reduce((first, second) => if (first._2 >= second._2) first else second)._1.getAs[Vector](dataColumn)
  }
}
