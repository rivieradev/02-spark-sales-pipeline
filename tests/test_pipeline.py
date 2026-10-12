from collections import Counter
from datetime import date
from decimal import Decimal

from main import (
    ORDER_SCHEMA,
    calculate_daily_revenue,
    validate_orders,
)

def make_order(
    order_id,
    *,
    quantity=1,
    unit_price="50.00",
    status="Completed",
    country="Greece"
):
    return (
        order_id,
        date(2026, 1, 1),
        "Keyboard",
        quantity,
        Decimal(unit_price),
        status,
        country
    )

def test_rejects_dupicates_and_invalid_quantity(spark):
    orders = spark.createDataFrame(
        [
            make_order(1),
            make_order(1),
            make_order(2),
            make_order(3, quantity=-1),
        ],
        schema=ORDER_SCHEMA
    )

    orders.select("order_id", "quantity").show()

    valid, rejected = validate_orders(orders)

    valid_ids = [
        row.order_id
        for row in valid.select("order_id").collect()
    ]

    rejected_rows = rejected.select(
        "order_id",
        "rejection_reason",
    ).collect()

    assert valid_ids == [2]

    assert Counter(
        row.order_id for row in rejected_rows
    ) == Counter({1: 2, 3: 1})

    duplicate_rows = [
        row
        for row in rejected_rows
        if row.order_id == 1
    ]

    assert all (
        "duplicate_order_id" in row.rejection_reason
        for row in duplicate_rows
    )

def test_calculated_completed_revenue(spark):
    orders = spark.createDataFrame(
        [
            make_order(
                1,
                quantity=2,
                unit_price="50.00",
            ),
            make_order(
                2,
                quantity=1,
                unit_price="25.00",
                status="Cancelled",
            ),
        ],
        schema=ORDER_SCHEMA
    )

    result = calculate_daily_revenue(orders).collect()

    assert len(result) == 1
    assert result[0].order_date == date(2026, 1, 1)
    assert result[0].customer_country == "Greece"
    assert result[0].revenue == Decimal("100.00")