package org.apache.spark.ml.made

import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}

trait StandardScalerParams extends HasInputCol with HasOutputCol {
  def setInputCol(value: String) : this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}

class StandardScaler(override val uid: String) extends Estimator[StandardScalerModel] with StandardScalerParams
with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("standardScaler"))

  override def fit(dataset: Dataset[_]): StandardScalerModel = {

    // Used to convert untyped dataframes to datasets with vectors
    implicit val encoder : Encoder[Vector] = ExpressionEncoder()

    val vectors: Dataset[Vector] = dataset.select(dataset($(inputCol)).as[Vector])

//    val dim: Int = AttributeGroup.fromStructField((dataset.schema($(inputCol)))).numAttributes.getOrElse(
//      vectors.first().size
//    )

    val summary = vectors.rdd.mapPartitions((data: Iterator[Vector]) => {
      val summarizer = new MultivariateOnlineSummarizer()
      data.foreach(v => summarizer.add(mllib.linalg.Vectors.fromBreeze(v.asBreeze)))
      Iterator(summarizer)
    }).reduce(_ merge _)

    copyValues(new StandardScalerModel(
      summary.mean.asML,
      Vectors.fromBreeze(breeze.numerics.sqrt(summary.variance.asBreeze)))).setParent(this)
//
//    val Row(row: Row) =  dataset
//      .select(Summarizer.metrics("mean", "std").summary(dataset($(inputCol))))
//      .first()
//
//    copyValues(new StandardScalerModel(row.getAs[Vector](0).toDense, row.getAs[Vector](1).toDense)).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[StandardScalerModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object StandardScaler extends DefaultParamsReadable[StandardScaler]

class StandardScalerModel private[made](
                           override val uid: String,
                           val means: DenseVector,
                           val stds: DenseVector) extends Model[StandardScalerModel] with StandardScalerParams with MLWritable {


  private[made] def this(means: Vector, stds: Vector) =
    this(Identifiable.randomUID("standardScalerModel"), means.toDense, stds.toDense)

  override def copy(extra: ParamMap): StandardScalerModel = copyValues(new StandardScalerModel(means, stds))

  override def transform(dataset: Dataset[_]): DataFrame = {
    val bMean = means.asBreeze
    val bStds = stds.asBreeze
    val transformUdf = dataset.sqlContext.udf.register(uid + "_transform",
      (x : Vector) => {
        Vectors.fromBreeze((x.asBreeze - bMean) /:/ bStds)
      })

    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val vectors = means.asInstanceOf[Vector] -> stds.asInstanceOf[Vector]

      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
    }
  }
}

object StandardScalerModel extends MLReadable[StandardScalerModel] {
  override def read: MLReader[StandardScalerModel] = new MLReader[StandardScalerModel] {
    override def load(path: String): StandardScalerModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      // Used to convert untyped dataframes to datasets with vectors
      implicit val encoder : Encoder[Vector] = ExpressionEncoder()

      val (means, std) =  vectors.select(vectors("_1").as[Vector], vectors("_2").as[Vector]).first()

      val model = new StandardScalerModel(means, std)
      metadata.getAndSetParams(model)
      model
    }
  }
}
