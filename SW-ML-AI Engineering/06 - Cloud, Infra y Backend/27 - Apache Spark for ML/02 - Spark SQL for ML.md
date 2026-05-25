# 📊 Spark SQL and ML Data Preparation

## Introduction

Feature engineering is where ML engineers spend 70-80% of their time — and it's where Spark delivers its most decisive advantage. A feature transformation that takes 30 minutes in pandas on a 5GB dataset can take 30 seconds in Spark on a 5TB dataset, simply because the work is distributed across 100+ cores instead of churning on one.

Spark SQL is the bridge between raw data and trained models. It provides a unified interface — you can write SQL queries, DataFrame transformations, or a mix of both — and Spark's Catalyst optimizer will produce the same efficient execution plan. This module covers the patterns, pitfalls, and performance optimizations specific to preparing ML datasets at scale.

---

## 1. 🔬 Spark SQL Fundamentals

Spark SQL provides two equivalent interfaces for the same engine:

```python
# SQL interface
spark.sql("""
    SELECT user_id, AVG(amount) as avg_spend, COUNT(*) as tx_count
    FROM transactions
    WHERE amount > 0
    GROUP BY user_id
""")

# DataFrame API (identical execution plan)
transactions \
    .filter(col("amount") > 0) \
    .groupBy("user_id") \
    .agg(
        avg("amount").alias("avg_spend"),
        count("*").alias("tx_count")
    )
```

### SQL vs DataFrame: When to Use Each

| Criterion | SQL | DataFrame API |
|---|---|---|
| **Readability** | Better for complex joins and aggregations | Better for chained transformations |
| **Dynamic queries** | Easier with string interpolation | Requires programmatic column references |
| **Type safety** | Runtime errors (string-based) | Compile-time in Scala, runtime in PySpark |
| **IDE support** | No autocomplete | Full autocomplete for column names/methods |
| **ML pipeline integration** | Requires temp views as intermediate step | Native chaining: `.transform()` → `.fit()` → `.transform()` |
| **Team accessibility** | Data analysts/SQL experts can contribute | ML engineers prefer programmatic style |

### Temporary Views for Hybrid Workflows

```python
# Create view for SQL access
transactions.createOrReplaceTempView("tx")

# Data analyst writes SQL
spark.sql("""
    CREATE OR REPLACE TEMP VIEW user_spend AS
    SELECT user_id, SUM(amount) AS total_spend, COUNT(*) AS tx_count
    FROM tx
    GROUP BY user_id
""")

# ML engineer continues with DataFrame API
user_spend = spark.table("user_spend")
features = user_spend.withColumn("avg_spend", col("total_spend") / col("tx_count"))
```

---

## 2. 🔄 ML Feature Engineering Patterns

### Pattern 1: Window Functions for Time-Based Features

Time-based features are the most common ML feature engineering pattern, and Spark's window functions make them efficient:

```python
from pyspark.sql.window import Window
from pyspark.sql.functions import lag, lead, avg, sum as spark_sum, row_number, datediff

# Define window: per user, ordered by time
user_window = Window.partitionBy("user_id").orderBy("timestamp")

time_features = (
    transactions
    .withColumn("prev_amount", lag("amount", 1).over(user_window))
    .withColumn("amount_change", col("amount") - col("prev_amount"))
    .withColumn("rolling_avg_7d", avg("amount").over(
        user_window.rowsBetween(-6, Window.currentRow)  # Last 7 rows
    ))
    .withColumn("rolling_sum_30d", spark_sum("amount").over(
        user_window.rowsBetween(-29, Window.currentRow)  # Last 30 rows
    ))
    .withColumn("days_since_last_tx", datediff(
        col("timestamp"),
        lag("timestamp", 1).over(user_window)
    ))
    .withColumn("tx_number", row_number().over(user_window))  # 1st, 2nd, 3rd tx
)
```

### Pattern 2: Categorical Encoding at Scale

| Method | Spark Implementation | When to Use |
|---|---|---|
| **One-Hot Encoding** | `OneHotEncoderEstimator` in ML pipeline | Low-cardinality categoricals (<50 categories) |
| **String Indexing** | `StringIndexer` in ML pipeline | Required step before one-hot encoding |
| **Target Encoding** | Manual groupBy + join (see below) | High-cardinality categoricals where mean encoding helps |
| **Hashing Trick** | `FeatureHasher` in ML pipeline | Very high cardinality (>10K), unknown categories at inference |
| **Frequency Encoding** | `groupBy + count + join` | Simple, interpretable replacement for high cardinality |

```python
# Target encoding: replace category with mean target per category
target_encoding = (
    transactions
    .groupBy("merchant_category")
    .agg(avg("is_fraud").alias("category_fraud_rate"))
)

# Join back to original features
features = transactions.join(target_encoding, "merchant_category", "left")
```

### Pattern 3: Handling Missing Values

```python
from pyspark.ml.feature import Imputer

# Numerical imputation
num_imputer = Imputer(
    inputCols=["amount", "quantity", "price"],
    outputCols=["amount_imp", "quantity_imp", "price_imp"],
    strategy="median"  # or "mean", "mode"
)

# Categorical imputation: add "missing" as a new category
features = features.fillna({"merchant_category": "UNKNOWN", "country": "UNKNOWN"})

# Advanced: impute with group median (per user)
user_medians = (
    transactions
    .groupBy("user_id")
    .agg(avg("amount").alias("user_avg_amount"))
)
features = transactions.join(user_medians, "user_id")
features = features.withColumn(
    "amount_filled",
    when(col("amount").isNull(), col("user_avg_amount")).otherwise(col("amount"))
)
```

---

## 3. 🏗️ The ML Pipeline Pattern

Spark MLlib provides a `Pipeline` class that chains transformers and estimators, ensuring consistent preprocessing across training and inference:

```mermaid
graph LR
    A[Raw DataFrame] --> B[StringIndexer<br/>Categorical → Index]
    B --> C[OneHotEncoder<br/>Index → Sparse Vector]
    C --> D[VectorAssembler<br/>Columns → Feature Vector]
    D --> E[Normalizer / StandardScaler<br/>Scale Features]
    E --> F[Model.fit()]
    F --> G[PipelineModel<br/>Can be saved, loaded, applied]
```

### Full Feature Engineering Pipeline

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, Imputer
)

# Step 1: Index categorical columns
merchant_indexer = StringIndexer(
    inputCol="merchant_category", outputCol="merchant_idx",
    handleInvalid="keep"  # New categories at inference → special index
)

# Step 2: One-hot encode indexed columns
encoder = OneHotEncoder(
    inputCols=["merchant_idx", "country_idx"],
    outputCols=["merchant_vec", "country_vec"]
)

# Step 3: Impute missing values
imputer = Imputer(
    inputCols=["amount", "quantity"],
    outputCols=["amount_imp", "quantity_imp"],
    strategy="median"
)

# Step 4: Assemble all feature columns into a single vector
assembler = VectorAssembler(
    inputCols=["amount_imp", "quantity_imp", "tx_count",
               "merchant_vec", "country_vec", "rolling_avg_7d"],
    outputCol="raw_features"
)

# Step 5: Scale features (critical for linear models, neural nets)
scaler = StandardScaler(
    inputCol="raw_features", outputCol="features",
    withStd=True, withMean=True
)

# Build pipeline
pipeline = Pipeline(stages=[
    merchant_indexer, encoder, imputer, assembler, scaler
])

# Fit on training data, transform both train and test
pipeline_model = pipeline.fit(train_df)
train_features = pipeline_model.transform(train_df)
test_features = pipeline_model.transform(test_df)

# Save pipeline for inference
pipeline_model.save("s3://models/feature_pipeline/")
```

---

## 4. ⚡ Performance Optimization for ML Workloads

### Partitioning Strategies

| Strategy | How | When |
|---|---|---|
| **Hash Partitioning** | `df.repartition(N, "key")` | Joins and aggregations on this key |
| **Round-Robin Partitioning** | `df.repartition(N)` | Balanced distribution, no key skew |
| **Coalesce** | `df.coalesce(N)` | Reducing partitions without shuffle (N < current) |
| **Pre-Partition on Write** | `df.write.partitionBy("date").parquet(...)` | Time-based partitioning for incremental reads |

### Caching Decisions

```python
from pyspark.storagelevel import StorageLevel

# Cache features DataFrame used across multiple model training iterations
features = spark.sql("SELECT ... complex feature query ...")
features.persist(StorageLevel.MEMORY_AND_DISK)  # Spill to disk if RAM is full

# Use cached features for:
rf_model = RandomForestClassifier().fit(features)     # Training 1
lr_model = LogisticRegression().fit(features)         # Training 2
gb_model = GBTClassifier().fit(features)              # Training 3

features.unpersist()  # Free memory when done
```

### Broadcast Join for Dimension Tables

```python
from pyspark.sql.functions import broadcast

# Large fact table (1TB) joined with small dimension table (10MB)
user_metadata = spark.read.parquet("s3://dim/users/")

# Broadcast hint: sends user_metadata to every executor
# Avoids shuffling the 1TB fact table
joined = transactions.join(broadcast(user_metadata), "user_id", "left")
```

---

## 5. 📊 EDA at Scale with Spark

### Summary Statistics for ML

```python
from pyspark.ml.stat import Correlation, ChiSquareTest

# Descriptive statistics
transactions.describe(["amount", "quantity", "is_fraud"]).show()

# Correlation matrix (Pearson)
correlation_matrix = Correlation.corr(
    feature_df, "features", method="pearson"
)

# Feature importance via Chi-Square test (categorical vs label)
chi_result = ChiSquareTest.test(
    feature_df, "features", "label"
)

# Quantile discretization for histogram features
from pyspark.ml.feature import QuantileDiscretizer
discretizer = QuantileDiscretizer(
    inputCol="amount", outputCol="amount_bucket",
    numBuckets=10
)
```

### Data Quality at Scale

```python
# Null ratio per column
null_stats = (
    transactions
    .select([
        (count(when(col(c).isNull(), c)) / count("*")).alias(f"{c}_null_ratio")
        for c in transactions.columns
    ])
)

# Cardinality check (too many unique values → might be an ID, not a feature)
cardinality = (
    transactions
    .agg(*[
        countDistinct(c).alias(f"{c}_cardinality")
        for c in ["user_id", "merchant_category", "event_type"]
    ])
)

# Distribution skew (is one category dominating?)
distribution = (
    transactions
    .groupBy("merchant_category")
    .agg(count("*").alias("cnt"))
    .withColumn("ratio", col("cnt") / transactions.count())
    .orderBy(col("cnt").desc())
)
```

---

## 6. 🌉 From Spark to Python ML (and Back)

Spark is for big data preparation, but many ML libraries (XGBoost, PyTorch, scikit-learn) work best on single-node data. The bridge:

### Spark → Pandas (for scikit-learn/XGBoost on sampled/small data)

```python
# 1. Aggregate features in Spark (handles TBs)
features = spark.sql("""
    SELECT user_id, AVG(amount), COUNT(*), MAX(amount) - MIN(amount)
    FROM transactions
    GROUP BY user_id
""")

# 2. Convert to pandas only AFTER aggregation (now manageable size)
features_pd = features.toPandas()

# 3. Train in scikit-learn
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier().fit(features_pd.drop("label", axis=1), features_pd["label"])
```

### Spark → Petastorm (for PyTorch distributed training on terabytes)

```python
# Spark prepares data as Parquet
spark.sql("SELECT features, label FROM training_set") \
    .write.mode("overwrite").parquet("s3://data/training/")

# PyTorch reads with Petastorm (lazy, batched, GPU-friendly)
from petastorm.pytorch import DataLoader
loader = DataLoader(reader, batch_size=256, shuffling_queue_capacity=4096)
```

---

## ⚠️ Pitfalls

- **`toPandas()` kills the Driver:** Only convert to pandas after aggregation has reduced the dataset to fit in Driver RAM. If your Spark DataFrame has 100M rows after aggregation, it will crash.
- **Skewed joins blow up tasks:** If one `user_id` has 1M transactions and the median has 10, the task processing that heavy key becomes the bottleneck. Use salting or adaptive query execution (AQE) to mitigate.
- **Lazy evaluation hides errors:** A typo in a column name won't error until an action is called. This can cause late-stage pipeline failures. Use `df.explain()` to verify the logical plan before executing.
- **SQL injection in dynamic queries:** If you build SQL strings with user input, use parameterized queries. Never use f-strings with untrusted input in Spark SQL.

---

## 💡 Tips

- **Enable Adaptive Query Execution (AQE):** `spark.sql.adaptive.enabled=true` — Spark 3.0+ dynamically optimizes shuffle partitions and join strategies at runtime based on actual data sizes.
- **Use `explain()` to debug:** `df.explain(extended=True)` shows the full Catalyst plan. Look for "Exchange" nodes (shuffles) and "Scan" operations to identify bottlenecks.
- **Prefer Delta Lake writes:** `df.write.format("delta").save(path)` over `df.write.parquet(path)`. Delta gives you ACID, time travel, and schema evolution for the same storage cost.
- **Vectorize UDFs with pandas UDF:** For custom transformations, use `@pandas_udf` instead of regular UDFs. Pandas UDFs process data in Arrow-optimized batches (10-100x faster).

---

## 📦 Compression Code

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, when, stddev
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler

spark = SparkSession.builder.appName("FeatureEngineering").getOrCreate()

# Load from Delta Lake
df = spark.read.format("delta").load("s3://bucket/events/")

# Feature engineering in Spark SQL
df.createOrReplaceTempView("events")
features = spark.sql("""
    SELECT
        user_id,
        AVG(amount) AS avg_amount,
        STDDEV(amount) AS std_amount,
        COUNT(*) AS event_count,
        COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) AS purchase_count,
        DATEDIFF(MAX(timestamp), MIN(timestamp)) AS activity_days
    FROM events
    WHERE amount > 0 AND timestamp IS NOT NULL
    GROUP BY user_id
""")

# Assemble into ML-ready feature vector
assembler = VectorAssembler(
    inputCols=["avg_amount", "std_amount", "event_count",
               "purchase_count", "activity_days"],
    outputCol="unscaled_features"
)
scaler = StandardScaler(inputCol="unscaled_features", outputCol="features",
                         withStd=True, withMean=True)

# Pipeline
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[assembler, scaler])
model = pipeline.fit(features)
result = model.transform(features)

result.select("user_id", "features").show(5)
print(f"Feature dimension: {result.select('features').first()[0].size}")
```

---

## ✅ Knowledge Check

1. **Why would you use SQL instead of the DataFrame API for feature engineering?** — SQL is more readable for complex joins/aggregations, accessible to data analysts who don't write Python, and identical in performance (both produce the same Catalyst execution plan).

2. **What does `VectorAssembler` do and why is it required for MLlib?** — It combines multiple feature columns into a single `features` vector column. MLlib's algorithms expect input as a single vector column, not separate columns.

3. **How do you handle new categorical values at inference time?** — `StringIndexer(handleInvalid="keep")` assigns unknown categories to a special index, preventing pipeline failure when new values appear in production that weren't in the training set.

4. **Why shouldn't you call `toPandas()` on a large Spark DataFrame?** — `toPandas()` transfers all data to the Driver node. If the DataFrame is larger than the Driver's RAM, it causes an OutOfMemoryError. Only convert after reducing data via aggregation or sampling.

---

## 🎯 Key Takeaways

- Spark SQL and DataFrame API produce identical Catalyst execution plans — use whichever is more readable for the task.
- Window functions are the most powerful tool for time-series feature engineering in Spark.
- ML Pipelines chain transformers (preprocessing) and estimators (training) into reproducible, inference-ready artifacts.
- Enable AQE for automatic runtime optimization; use broadcast hints for joining large fact tables with small dimension tables.
- Spark is for big data prep — aggregate in Spark, then convert to pandas/PyTorch for single-node training.

---

## References

- [Spark SQL Programming Guide](https://spark.apache.org/docs/latest/sql-programming-guide.html)
- [ML Pipelines in Spark](https://spark.apache.org/docs/latest/ml-pipeline.html)
- [Spark Window Functions](https://spark.apache.org/docs/latest/sql-ref-syntax-qry-select-window.html)
- [Adaptive Query Execution](https://spark.apache.org/docs/latest/sql-performance-tuning.html#adaptive-query-execution)
