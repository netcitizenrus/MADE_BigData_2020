package org.apache.spark.ml.made

import org.apache.spark.sql.SparkSession

trait WithSpark {
   lazy val spark = WithSpark._spark
   lazy val sqlc = WithSpark._sqlc
}

object WithSpark {
  lazy val _spark = SparkSession.builder
    .appName("Simple Application")
    .master("local[4]")
    .getOrCreate()

  lazy val _sqlc = _spark.sqlContext
}
