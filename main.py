from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import DataFrame, Window

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

def validate_orders(
    orders: DataFrame,
) -> tuple[DataFrame, DataFrame]:
    order_id_window = Window.partitionBy("order_id")

    validated = orders.withColumn(
        "_order_id_count",
        F.count("*").over(order_id_window),
    )

    duplicate_condition = (
        F.col("order_id").isNotNull()
        & (F.col("_order_id_count") > 1)
    )

    validated = validated.withColumn(
        "rejection_reason",
        F.concat_ws(
            ", ",
            F.when(
                F.col("order_id").isNull(),
                F.lit("missing_order_id"),
            ),
            F.when(
                duplicate_condition,
                F.lit("duplicate_order_id"),
            ),
            F.when(
                F.col("order_date").isNull(),
                F.lit("missing_order_date"),
            ),
            F.when(
                F.col("quantity").isNull(),
                F.lit("missing_quantity"),
            ),
            F.when(
                F.col("quantity") <= 0,
                F.lit("quantity_must_be_positive"),
            ),
            F.when(
                F.col("unit_price").isNull(),
                F.lit("missign_unit_price"),
            ),
            F.when(
                F.col("unit_price") < 0,
                F.lit("unit_price_must_be_non_negative"),
            ),
            F.when(
                F.col("customer_country").isNull()
                | (F.trim(F.col("customer_country")) == ""),
                F.lit("missing_customer_country")
            ),
        ),
    )

    valid_orders = (
        validated
        .filter(F.col("rejection_reason") == "")
        .drop("_order_id_count", "rejection_reason")
    )

    rejected_orders = (
        validated
        .filter(F.col("rejection_reason") != "")
        .drop("_order_id_count")
    )

    return valid_orders, rejected_orders

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

        valid_orders, rejected_orders = validate_orders(orders)

        print("Input schema:")
        orders.printSchema()

        print("Rejected orders:")
        rejected_orders.show(truncate=False)

        completed_orders = valid_orders.filter(
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

        (
            valid_orders.write
            .mode("overwrite")
            .parquet("data/output/valid_orders")
        )

        (
            rejected_orders.write
            .mode("overwrite")
            .parquet("data/output/rejected_orders")
        )
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
