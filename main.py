from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

INPUT_FILE = Path("data/input/orders.csv")
OUTPUT_DIRECTORY = Path("data/output/daily_revenue")

ORDER_SCHEMA = T.StructType(
    [
        T.StructField("order_id", T.LongType(), nullable=True),
        T.StructField("order_date", T.DateType(), nullable=True),
        T.StructField("product", T.StringType(), nullable=True),
        T.StructField("quantity", T.IntegerType(), nullable=True),
        T.StructField(
            "unit_price",
            T.DecimalType(10, 2),
            nullable=True
        ),
        T.StructField("status", T.StringType(), nullable=True),
        T.StructField(
            "customer_country",
            T.StringType(),
            nullable=True
        ),
    ]
)

def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .master("local[*]") # use all available local CPU
        .appName("spark-sales-pipeline")
        .getOrCreate()
    )

def main() -> None:
    spark = create_spark_session()

    try:
        orders = (
            spark.read
            .option("header", True)
            .option("dateFormat", "yyyy-MM-dd")
            .schema(ORDER_SCHEMA)
            .csv(str(INPUT_FILE))
        )

        print("Input schema:")
        orders.printSchema()

        completed_orders = orders.filter(
            F.col("status") == "Completed"
        )

        orders_with_total = completed_orders.withColumn(
            "line_total",
            F.round(
                F.col("quantity") * F.col("unit_price"),
                2,
            ),
        )

        daily_revenue = (
            orders_with_total
            .groupBy("order_date", "customer_country")
            .agg(
                F.sum("line_total").alias("revenue")
            )
            .orderBy("order_date", "customer_country")
        )

        print("Execution plan:")
        daily_revenue.explain(mode="formatted")

        print("Daily revenue:")
        daily_revenue.show(truncate=False)
        (
            daily_revenue.write
            .mode("overwrite")
            .parquet(str(OUTPUT_DIRECTORY))
        )
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
