import pytest
from pyspark.sql import SparkSession

@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("spark-sales-test")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    yield session

    session.stop()