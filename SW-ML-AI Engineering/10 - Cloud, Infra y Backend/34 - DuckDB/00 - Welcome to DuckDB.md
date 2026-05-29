# 🦆 Welcome to DuckDB

## 🎯 Learning Objectives
- Understand what DuckDB is and where it fits in the data engineering landscape
- Contrast DuckDB with pandas, Spark, PostgreSQL, and SQLite
- Recognize the OLAP/OLTP distinction and why columnar execution matters
- Map the 3-note course structure and prerequisites

## Introduction

**Etymology.** The name "DuckDB" comes from the animal kingdom, not the tech world. A duck is small, agile, and surprisingly powerful — it navigates land, water, and air with ease. DuckDB is the same: it runs embedded inside a Python, Go, or Rust process, handles analytical queries on GB-scale data with sub-second latency, and requires zero configuration. Unlike the heavyweight "elephants" (PostgreSQL, Spark) or "whales" (BigQuery clusters), a duck fits wherever your code runs. There is no server process, no port to open, no YAML config file at `/etc/duckdb.yaml`. Just `pip install duckdb` and you have a full OLAP engine inside your Python runtime.

**The Problem.** Data scientists love pandas for exploration on datasets under 1 GB. But real-world analytics routinely hit 10 GB, 50 GB, or 500 GB. At that scale pandas' in-memory model collapses with an out-of-memory kill, and Apache Spark becomes the default answer — but Spark requires a cluster, JVM tuning, and YAML ceremonies that would distract any ML engineer. Between the "too small" (pandas) and "too big" (Spark) extremes sits DuckDB: an embedded columnar database that runs analytical SQL at 10-100x pandas speed on the same laptop, with automatic spilling to disk when data exceeds RAM. It is the database that ML engineers never knew they needed.

**Course Map.** This course has 4 notes. Note 00 (you are here) provides context and orientation. [[01 - DuckDB Fundamentals - In-Process OLAP with SQL]] covers the core engine: columnar storage, vectorized execution, the full SQL feature set, and direct Parquet/CSV/JSON queries. [[02 - DuckDB with Python - DataFrames, Parquet and SQL Integration]] explores the Python API and the zero-copy Arrow interop with pandas and Polars. [[03 - DuckDB in ML Pipelines - RAG Preprocessing, Feature Engineering and Production]] applies DuckDB to RAG document preprocessing, feature engineering pipelines, and production analytics sidecars. Prerequisites: working knowledge of SQL ([[01 - Curso SQL con PostgreSQL]]), basic Python, and familiarity with pandas DataFrames.

---

## 1. The OLAP Spectrum — Where DuckDB Lands

| Tool | Paradigm | Scale sweet spot | Setup cost |
|------|-----------|------------------|------------|
| pandas | In-memory DataFrame | < 1 GB | `pip install pandas` |
| **DuckDB** | **Embedded columnar DB** | **1 GB – 500 GB** | **`pip install duckdb`** |
| Polars | In-memory/streaming DataFrame | 1 GB – 100 GB | `pip install polars` |
| Spark | Distributed cluster | > 100 GB | Cluster config + JVM |
| BigQuery | Serverless cloud OLAP | TB – PB | Cloud project + billing |

DuckDB carves out the middle: it handles what pandas cannot and what Spark would overengineer.

## 2. SQLite for OLTP, DuckDB for OLAP

SQLite proved that a zero-config, serverless, embedded database can power billions of mobile apps and browsers. DuckDB applies the same philosophy to analytical workloads. The key difference: SQLite stores rows contiguously (good for point lookups: `SELECT * FROM users WHERE id = 42`), while DuckDB stores columns contiguously (good for aggregations: `SELECT country, AVG(age) FROM population GROUP BY country`). A columnar layout lets DuckDB skip irrelevant columns entirely, apply compression schemes like dictionary encoding and RLE, and feed vectors of 2048 values at a time into CPU SIMD lanes. The result is a 10-50x speedup over row-based engines on analytical queries — all inside the same process as your Python code.

## 3. Zero Dependencies, Single-File Databases

DuckDB has no external dependencies. It compiles into a single library that you install via `pip install duckdb` or `go get github.com/marcboeker/go-duckdb`. A DuckDB database is a single `.duckdb` file (or `:memory:` for transient workloads), just like SQLite's `.sqlite` file. You can email a `sales.duckdb` file to a colleague and they can query it immediately — no `pg_dump`/`pg_restore` dance. For ML pipelines this means: preprocess data into a `.duckdb` file, ship it alongside your model checkpoint, and run ad-hoc analytical queries in production with no database service to maintain.

---

## 🎯 Key Takeaways
- DuckDB is an embedded, serverless, zero-dependency OLAP database — it runs inside your process.
- It occupies the gap between pandas (< 1 GB, in-memory) and Spark (> 100 GB, cluster).
- Columnar storage + vectorized execution gives 10-100x speedup on analytical queries versus row-based databases.
- DuckDB reads CSV, JSON, and Parquet files directly via SQL — no CREATE TABLE or ETL required.
- A `.duckdb` file is portable, self-contained, and deployable alongside any application.
- DuckDB is the database for ML engineers: preprocessing, feature engineering, and production analytics.
- This course: fundamentals → Python interop → ML pipeline integration.

## 📦 Código de Compresión

```python
# duckdb_welcome_demo.py — Run as: pip install duckdb pandas && python duckdb_welcome_demo.py
import duckdb

# Create an in-memory database — no server, no config, no YAML
conn = duckdb.connect()

# Query CSV directly without imports — DuckDB auto-infers schema from the first 20K rows
result = conn.sql("""
    SELECT species, COUNT(*) AS count, ROUND(AVG(sepal_length), 2) AS avg_sepal_length
    FROM 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'
    GROUP BY species
    ORDER BY count DESC
""")
print(result)  # ¡Sorpresa! 150 rows grouped in < 5ms — no pandas, no Spark, just SQL in-process

# Same result as pandas DataFrame with .df()
df = result.df()
print(f"\nDataFrame shape: {df.shape}")

# Close — or let the process exit, it doesn't matter (embedded, no server)
conn.close()
```

## References
- [DuckDB Official Documentation](https://duckdb.org/docs/)
- [DuckDB Architecture Paper (CIDR 2022)](https://duckdb.org/why_duckdb)
- [MotherDuck — Managed DuckDB Cloud](https://motherduck.com/)
- [[01 - DuckDB Fundamentals - In-Process OLAP with SQL]]
- [[01 - Curso SQL con PostgreSQL]]
- [[06/27 - Apache Spark for ML]]
- [[14/03 - Rust Polars Internals]]
- [[10/28 - BigQuery for ML]]
