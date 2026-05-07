# ⚙️ High-Performance ETL Pipelines in Rust

## Introduction

High-performance ETL (Extract, Transform, Load) pipelines in Rust represent a paradigm shift from traditional Python or Java-based data processing. Rust's memory safety guarantees, zero-cost abstractions, and fearless concurrency enable building data pipelines that can process billions of records per day on a single machine, eliminating the need for complex distributed systems for many use cases. Real case: **Instacart** rebuilt their real-time inventory ETL system in Rust, reducing their processing cluster from 50 nodes to 5 nodes while improving throughput by 10x.

The async ecosystem in Rust, powered by tokio and async-stream, provides the foundation for building non-blocking, high-throughput data pipelines. Unlike synchronous ETL tools that block on I/O operations, Rust's async model allows overlapping network requests, disk I/O, and CPU-bound transformations. Combined with backpressure mechanisms and bounded channels, this creates pipelines that gracefully handle variable data rates without memory exhaustion.

⚠️ **Warning:** Async Rust has a steep learning curve. Common pitfalls include holding locks across `.await` points, creating unbounded channels that cause memory leaks, and accidentally creating blocking operations that stall the entire async runtime. Always use `tokio::task::spawn_blocking` for CPU-intensive or blocking I/O operations.

💡 **Tip:** Use `async-stream` crate for ergonomic stream creation. It provides a `stream!` macro that lets you write async generators similar to Python's async generators, making pipeline composition much more readable.

## 1. Pipeline Architecture

ETL pipeline architecture in Rust follows a modular design pattern with clear separation between extraction, transformation, and loading phases. Each phase is implemented as a stream that can be composed together.

**Core Components:**
- **Source**: Async reader that pulls data from various sources (files, databases, APIs)
- **Transform**: Stream processor that applies business logic, filtering, and enrichment
- **Sink**: Async writer that pushes processed data to destinations
- **Router**: Optional component for splitting/merging data streams

**Design Patterns:**
1. **Fan-out/Fan-in**: Parallel processing across multiple worker threads
2. **Pipeline Parallelism**: Different stages run concurrently on different data batches
3. **Backpressure**: Using bounded channels to prevent memory exhaustion
4. **Exactly-Once Processing**: Idempotent operations with checkpointing

Real case: **DoorDash** uses Rust-based ETL pipelines for real-time delivery optimization, processing GPS coordinates from thousands of drivers with sub-second latency.

## 2. Async ETL with Tokio

Tokio provides the async runtime foundation for building high-performance ETL pipelines. Combined with `futures` and `async-stream`, it enables building complex data flows with minimal boilerplate.

**Key Tokio Features for ETL:**
- **Multi-threaded runtime**: Automatic work stealing across CPU cores
- **Bounded channels**: Backpressure support between pipeline stages
- **Task spawning**: Lightweight concurrent tasks for parallel processing
- **I/O drivers**: Efficient async I/O for files, networks, and databases

**Comparison of ETL Frameworks:**
| Framework | Language | Throughput | Latency | Memory | Complexity |
|-----------|----------|------------|---------|---------|------------|
| Apache Spark | Scala/Java | High | Seconds | High | Very High |
| Pandas | Python | Medium | Seconds | High | Medium |
| Polars | Rust/Python | Very High | Milliseconds | Low | Medium |
| Custom Tokio | Rust | Extremely High | Microseconds | Very Low | High |
| DataFusion | Rust | Very High | Milliseconds | Low | Medium |

**Formula for pipeline throughput:**
```
Throughput = Records_Processed / Time
Effective_Throughput = min(Throughput_Source, Throughput_Transform, Throughput_Sink)
```

⚠️ **Warning:** Tokio's default multi-threaded runtime can cause issues with thread-local storage. If your transformations use thread-local data (like random number generators), you must use `tokio::task::spawn_local` on a single-threaded runtime.

💡 **Tip:** Use `tokio::stream::StreamExt` combinator methods (`map`, `filter`, `fold`) for concise stream processing. They're more readable than manual `poll_next` implementations.

## 3. Transformations and Window Functions

Rust provides powerful primitives for implementing complex data transformations, from simple mapping to advanced window functions used in time-series analysis.

**Common Transformations:**
- **Map**: Apply function to each record (1:1 transformation)
- **Filter**: Select records matching criteria (1:0+ transformation)
- **FlatMap**: Expand records into multiple outputs (1:N transformation)
- **Aggregate**: Combine multiple records into summary (N:1 transformation)
- **Window**: Apply operations over sliding/tumbling windows

**Window Function Types:**
- **Tumbling Window**: Fixed-size, non-overlapping windows
- **Sliding Window**: Fixed-size, overlapping windows
- **Session Window**: Dynamic windows based on activity gaps
- **Global Window**: All data in a single window

```mermaid
graph TB
    subgraph "ETL Pipeline Architecture"
        A[Data Sources] --> B[Extractor]
        B --> C[Channel: bounded(1000)]
        C --> D[Transformer 1: Parse]
        D --> E[Channel: bounded(1000)]
        E --> F[Transformer 2: Validate]
        F --> G[Channel: bounded(1000)]
        G --> H[Transformer 3: Enrich]
        H --> I[Channel: bounded(1000)]
        I --> J[Sink: Write]
        J --> K[Destination]
        
        L[Error Handler] -.-> D
        L -.-> F
        L -.-> H
        
        M[Metrics Collector] -.-> D
        M -.-> F
        M -.-> H
    end
    
    style A fill:#e3f2fd
    style K fill:#f1f8e9
    style L fill:#ffebee
    style M fill:#fff3e0
```

## 4. Rust Code Examples

```rust
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use futures::stream;
use std::pin::Pin;
use std::time::Duration;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SensorReading {
    device_id: String,
    timestamp: u64,
    temperature: f64,
    humidity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProcessedReading {
    device_id: String,
    timestamp: u64,
    temp_fahrenheit: f64,
    comfort_index: f64,
    anomaly_score: f64,
}

// Basic ETL pipeline with error handling
async fn simple_etl_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    // Channel for passing data between stages
    let (tx, mut rx) = mpsc::channel(1000);
    
    // Spawn extractor task
    let extractor_handle = tokio::spawn(async move {
        for i in 0..100 {
            let reading = SensorReading {
                device_id: format!("device_{}", i % 10),
                timestamp: chrono::Utc::now().timestamp() as u64,
                temperature: 20.0 + (i as f64 * 0.1),
                humidity: 50.0 + (i as f64 * 0.05),
            };
            
            if tx.send(reading).await.is_err() {
                break; // Receiver dropped
            }
            
            // Simulate extraction delay
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });
    
    // Process stream
    let mut processed_count = 0;
    while let Some(reading) = rx.recv().await {
        let processed = transform_reading(reading);
        
        // Simulate sink operation
        sink_write(&processed).await?;
        processed_count += 1;
        
        if processed_count % 10 == 0 {
            println!("Processed {} readings", processed_count);
        }
    }
    
    extractor_handle.await?;
    println!("ETL pipeline complete. Total processed: {}", processed_count);
    Ok(())
}

fn transform_reading(reading: SensorReading) -> ProcessedReading {
    let temp_fahrenheit = reading.temperature * 9.0 / 5.0 + 32.0;
    
    // Simple comfort index calculation
    let comfort_index = if reading.temperature > 18.0 && reading.temperature < 26.0 
        && reading.humidity > 30.0 && reading.humidity < 70.0 {
        1.0
    } else {
        0.0
    };
    
    // Simple anomaly detection
    let anomaly_score = if reading.temperature > 35.0 || reading.temperature < 0.0 {
        1.0
    } else if reading.temperature > 30.0 || reading.temperature < 5.0 {
        0.5
    } else {
        0.0
    };
    
    ProcessedReading {
        device_id: reading.device_id,
        timestamp: reading.timestamp,
        temp_fahrenheit,
        comfort_index,
        anomaly_score,
    }
}

async fn sink_write(reading: &ProcessedReading) -> Result<(), Box<dyn std::error::Error>> {
    // Simulate async write to database/file
    tokio::time::sleep(Duration::from_millis(1)).await;
    Ok(())
}

// Advanced: Parallel processing with fan-out/fan-in
async fn parallel_etl_pipeline(workers: usize) -> Result<(), Box<dyn std::error::Error>> {
    let (source_tx, source_rx) = mpsc::channel(1000);
    let (sink_tx, mut sink_rx) = mpsc::channel(1000);
    
    // Spawn multiple transformer workers
    let mut worker_handles = Vec::new();
    for worker_id in 0..workers {
        let mut rx = source_rx.clone();
        let tx = sink_tx.clone();
        
        let handle = tokio::spawn(async move {
            while let Some(reading) = rx.recv().await {
                // Each worker processes readings
                let processed = transform_reading(reading);
                
                // Apply worker-specific logic
                let enhanced = if worker_id % 2 == 0 {
                    // Even workers add extra validation
                    let mut p = processed;
                    p.anomaly_score *= 1.1; // Slightly higher sensitivity
                    p
                } else {
                    processed
                };
                
                if tx.send(enhanced).await.is_err() {
                    break;
                }
            }
        });
        
        worker_handles.push(handle);
    }
    
    // Drop original channel sender to allow worker receivers to terminate
    drop(source_tx);
    
    // Spawn sink aggregator
    let sink_handle = tokio::spawn(async move {
        let mut batch = Vec::new();
        while let Some(reading) = sink_rx.recv().await {
            batch.push(reading);
            
            // Batch writes for efficiency
            if batch.len() >= 100 {
                println!("Writing batch of {} readings", batch.len());
                batch.clear();
            }
        }
        
        // Write remaining
        if !batch.is_empty() {
            println!("Writing final batch of {} readings", batch.len());
        }
    });
    
    // Wait for all workers
    for handle in worker_handles {
        handle.await?;
    }
    
    sink_handle.await?;
    Ok(())
}

// Stream-based pipeline with async-stream
async fn stream_etl_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    use async_stream::stream;
    
    // Create source stream
    let source_stream = stream! {
        for i in 0..1000 {
            yield SensorReading {
                device_id: format!("sensor_{:03}", i % 100),
                timestamp: chrono::Utc::now().timestamp() as u64,
                temperature: 20.0 + (i as f64 * 0.01).sin() * 10.0,
                humidity: 50.0 + (i as f64 * 0.02).cos() * 20.0,
            };
            
            if i % 100 == 0 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        }
    };
    
    // Process stream with combinators
    let mut processed_stream = Box::pin(source_stream
        .map(|reading| transform_reading(reading))
        .filter(|processed| {
            // Filter out normal readings, keep anomalies
            processed.anomaly_score > 0.0
        })
        .chunks(100) // Batch every 100 readings
    );
    
    let mut total_anomalies = 0;
    while let Some(batch) = processed_stream.next().await {
        println!("Processing batch of {} anomalies", batch.len());
        total_anomalies += batch.len();
        
        // Simulate batch write
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    
    println!("Total anomalies detected: {}", total_anomalies);
    Ok(())
}
```

```rust
// Window functions for time-series data
use std::time::{Duration, Instant};
use tokio::time;

struct WindowedAggregator {
    window_size: Duration,
    slide_size: Duration,
    buffers: std::collections::HashMap<String, Vec<(Instant, f64)>>,
}

impl WindowedAggregator {
    fn new(window_size: Duration, slide_size: Duration) -> Self {
        Self {
            window_size,
            slide_size,
            buffers: std::collections::HashMap::new(),
        }
    }
    
    fn add_reading(&mut self, device_id: String, value: f64) {
        let buffer = self.buffers.entry(device_id).or_insert_with(Vec::new);
        buffer.push((Instant::now(), value));
        
        // Remove old readings outside window
        let cutoff = Instant::now() - self.window_size;
        buffer.retain(|(timestamp, _)| *timestamp > cutoff);
    }
    
    fn get_window_average(&self, device_id: &str) -> Option<f64> {
        if let Some(buffer) = self.buffers.get(device_id) {
            let cutoff = Instant::now() - self.window_size;
            let recent: Vec<_> = buffer.iter()
                .filter(|(timestamp, _)| *timestamp > cutoff)
                .map(|(_, value)| *value)
                .collect();
            
            if recent.is_empty() {
                None
            } else {
                Some(recent.iter().sum::<f64>() / recent.len() as f64)
            }
        } else {
            None
        }
    }
    
    fn get_window_max(&self, device_id: &str) -> Option<f64> {
        if let Some(buffer) = self.buffers.get(device_id) {
            let cutoff = Instant::now() - self.window_size;
            buffer.iter()
                .filter(|(timestamp, _)| *timestamp > cutoff)
                .map(|(_, value)| *value)
                .fold(None, |acc, val| {
                    Some(match acc {
                        Some(max) => max.max(val),
                        None => val,
                    })
                })
        } else {
            None
        }
    }
}

// Tumbling window implementation
async fn tumbling_window_aggregation() -> Result<(), Box<dyn std::error::Error>> {
    let window_size = Duration::from_secs(5);
    let mut aggregator = WindowedAggregator::new(window_size, window_size);
    
    let mut interval = time::interval(Duration::from_millis(100));
    let mut readings_count = 0;
    
    for _ in 0..100 {
        interval.tick().await;
        
        // Simulate incoming readings
        let device_id = format!("sensor_{}", readings_count % 5);
        let value = 20.0 + (readings_count as f64 * 0.1).sin() * 10.0;
        
        aggregator.add_reading(device_id.clone(), value);
        readings_count += 1;
        
        // Every 5 seconds, output window results
        if readings_count % 50 == 0 {
            for device in 0..5 {
                let device_id = format!("sensor_{}", device);
                if let Some(avg) = aggregator.get_window_average(&device_id) {
                    println!("Device {}: Window average = {:.2}", device_id, avg);
                }
            }
        }
    }
    
    Ok(())
}
```

---

## 📦 Compression Code

```rust
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::io::Write;

/// High-performance async ETL pipeline with compression
/// Processes, compresses, and batches data for efficient storage
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("High-Performance Async ETL Pipeline");
    println!("===================================\n");
    
    // Pipeline configuration
    let config = PipelineConfig {
        batch_size: 1000,
        compression_level: 6,
        window_size_seconds: 5,
        max_inflight_batches: 10,
    };
    
    // Create bounded channels for backpressure
    let (raw_tx, raw_rx) = mpsc::channel(5000);
    let (processed_tx, processed_rx) = mpsc::channel(5000);
    let (compressed_tx, compressed_rx) = mpsc::channel(1000);
    
    // Start pipeline stages
    let extractor_handle = tokio::spawn(extractor_stage(raw_tx));
    let transformer_handle = tokio::spawn(transformer_stage(raw_rx, processed_tx));
    let compressor_handle = tokio::spawn(compressor_stage(processed_rx, compressed_tx, config.clone()));
    let sink_handle = tokio::spawn(sink_stage(compressed_rx, config));
    
    // Monitor pipeline metrics
    let monitor_handle = tokio::spawn(monitor_pipeline());
    
    // Wait for extractor to complete
    extractor_handle.await?;
    println!("Extraction complete");
    
    // Wait for transformer
    transformer_handle.await?;
    println!("Transformation complete");
    
    // Wait for compressor
    compressor_handle.await?;
    println!("Compression complete");
    
    // Wait for sink
    sink_handle.await?;
    println!("Sink complete");
    
    // Stop monitor
    monitor_handle.abort();
    
    println!("\nPipeline execution completed successfully");
    Ok(())
}

#[derive(Clone)]
struct PipelineConfig {
    batch_size: usize,
    compression_level: u32,
    window_size_seconds: u64,
    max_inflight_batches: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawData {
    id: u64,
    timestamp: u64,
    source: String,
    payload: String,
    size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProcessedData {
    id: u64,
    timestamp: u64,
    source: String,
    payload: String,
    size: usize,
    processed_at: u64,
    checksum: u32,
    metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
struct CompressedBatch {
    batch_id: u64,
    compressed_data: Vec<u8>,
    original_size: usize,
    compressed_size: usize,
    compression_ratio: f64,
    record_count: usize,
}

// Stage 1: Data extraction
async fn extractor_stage(tx: mpsc::Sender<RawData>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Extractor stage started");
    
    let mut id = 0;
    let start = Instant::now();
    
    // Simulate data extraction from multiple sources
    for source_num in 0..5 {
        let source = format!("source_{}", source_num);
        
        for i in 0..1000 {
            id += 1;
            
            let data = RawData {
                id,
                timestamp: chrono::Utc::now().timestamp() as u64,
                source: source.clone(),
                payload: format!("data_{}_{}_{}", source_num, i, uuid::Uuid::new_v4()),
                size: 100 + (i % 900), // Varying sizes
            };
            
            if tx.send(data).await.is_err() {
                break;
            }
            
            // Simulate extraction rate
            if i % 100 == 0 {
                tokio::time::sleep(Duration::from_micros(100)).await;
            }
        }
        
        println!("Extracted from {}", source);
    }
    
    let elapsed = start.elapsed();
    println!("Extractor completed: {} records in {:?}", id, elapsed);
    println!("Extraction rate: {:.0} records/sec", id as f64 / elapsed.as_secs_f64());
    
    Ok(())
}

// Stage 2: Data transformation
async fn transformer_stage(
    mut rx: mpsc::Receiver<RawData>,
    tx: mpsc::Sender<ProcessedData>
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Transformer stage started");
    
    let mut processed_count = 0;
    let start = Instant::now();
    
    while let Some(raw) = rx.recv().await {
        let processed = transform_data(raw).await?;
        
        if tx.send(processed).await.is_err() {
            break;
        }
        
        processed_count += 1;
        
        if processed_count % 1000 == 0 {
            let elapsed = start.elapsed();
            println!("Transformed {} records ({:.0} records/sec)", 
                     processed_count, 
                     processed_count as f64 / elapsed.as_secs_f64());
        }
    }
    
    let elapsed = start.elapsed();
    println!("Transformer completed: {} records in {:?}", processed_count, elapsed);
    
    Ok(())
}

async fn transform_data(raw: RawData) -> Result<ProcessedData, Box<dyn std::error::Error + Send + Sync>> {
    // Simulate CPU-intensive transformation
    let checksum = calculate_checksum(&raw.payload);
    
    let mut metadata = HashMap::new();
    metadata.insert("original_size".to_string(), raw.size.to_string());
    metadata.insert("transform_version".to_string(), "1.0".to_string());
    metadata.insert("source_type".to_string(), if raw.source.contains("api") { "api" } else { "file" }.to_string());
    
    Ok(ProcessedData {
        id: raw.id,
        timestamp: raw.timestamp,
        source: raw.source,
        payload: raw.payload,
        size: raw.size,
        processed_at: chrono::Utc::now().timestamp() as u64,
        checksum,
        metadata,
    })
}

fn calculate_checksum(data: &str) -> u32 {
    // Simple checksum calculation
    data.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32))
}

// Stage 3: Compression and batching
async fn compressor_stage(
    mut rx: mpsc::Receiver<ProcessedData>,
    tx: mpsc::Sender<CompressedBatch>,
    config: PipelineConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Compressor stage started");
    
    let mut batch = Vec::with_capacity(config.batch_size);
    let mut batch_id = 0;
    let mut total_records = 0;
    let mut total_original_size = 0;
    let mut total_compressed_size = 0;
    
    let start = Instant::now();
    
    while let Some(data) = rx.recv().await {
        batch.push(data);
        total_records += 1;
        total_original_size += batch.last().unwrap().size;
        
        if batch.len() >= config.batch_size {
            let compressed = compress_batch(&batch, batch_id, &config)?;
            total_compressed_size += compressed.compressed_size;
            
            if tx.send(compressed).await.is_err() {
                break;
            }
            
            batch.clear();
            batch_id += 1;
            
            if batch_id % 10 == 0 {
                let elapsed = start.elapsed();
                let avg_ratio = if total_original_size > 0 {
                    total_original_size as f64 / total_compressed_size as f64
                } else {
                    0.0
                };
                
                println!("Compressed {} batches ({} records, ratio: {:.2}:1)", 
                         batch_id, total_records, avg_ratio);
            }
        }
    }
    
    // Compress remaining records
    if !batch.is_empty() {
        let compressed = compress_batch(&batch, batch_id, &config)?;
        total_compressed_size += compressed.compressed_size;
        tx.send(compressed).await?;
        batch_id += 1;
    }
    
    let elapsed = start.elapsed();
    let final_ratio = if total_original_size > 0 {
        total_original_size as f64 / total_compressed_size as f64
    } else {
        0.0
    };
    
    println!("Compressor completed: {} batches, {} records", batch_id, total_records);
    println!("Total compression ratio: {:.2}:1", final_ratio);
    println!("Compression rate: {:.0} records/sec", total_records as f64 / elapsed.as_secs_f64());
    
    Ok(())
}

fn compress_batch(
    batch: &[ProcessedData],
    batch_id: u64,
    config: &PipelineConfig,
) -> Result<CompressedBatch, Box<dyn std::error::Error + Send + Sync>> {
    // Serialize batch to JSON
    let serialized = serde_json::to_vec(batch)
        .map_err(|e| format!("Serialization failed: {}", e))?;
    
    // Compress with GZIP
    let mut encoder = GzEncoder::new(Vec::new(), Compression::new(config.compression_level));
    encoder.write_all(&serialized)
        .map_err(|e| format!("Compression failed: {}", e))?;
    
    let compressed = encoder.finish()
        .map_err(|e| format!("Compression finish failed: {}", e))?;
    
    let original_size = serialized.len();
    let compressed_size = compressed.len();
    let compression_ratio = if compressed_size > 0 {
        original_size as f64 / compressed_size as f64
    } else {
        0.0
    };
    
    Ok(CompressedBatch {
        batch_id,
        compressed_data: compressed,
        original_size,
        compressed_size,
        compression_ratio,
        record_count: batch.len(),
    })
}

// Stage 4: Sink (write to storage)
async fn sink_stage(
    mut rx: mpsc::Receiver<CompressedBatch>,
    _config: PipelineConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Sink stage started");
    
    let mut total_batches = 0;
    let mut total_records = 0;
    let start = Instant::now();
    
    while let Some(batch) = rx.recv().await {
        // Simulate writing to storage (could be S3, local disk, database)
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        total_batches += 1;
        total_records += batch.record_count;
        
        if total_batches % 10 == 0 {
            let elapsed = start.elapsed();
            println!("Written {} batches ({} records)", total_batches, total_records);
            println!("Write rate: {:.0} records/sec", total_records as f64 / elapsed.as_secs_f64());
        }
    }
    
    let elapsed = start.elapsed();
    println!("Sink completed: {} batches, {} records", total_batches, total_records);
    println!("Total write time: {:?}", elapsed);
    
    Ok(())
}

// Monitor pipeline metrics
async fn monitor_pipeline() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Pipeline monitor started");
    
    let mut interval = tokio::time::interval(Duration::from_secs(5));
    let mut iteration = 0;
    
    loop {
        interval.tick().await;
        iteration += 1;
        
        // In a real implementation, you would collect metrics from each stage
        println!("[Monitor] Pipeline health check #{}", iteration);
        println!("[Monitor] - Memory: {} MB (estimated)", iteration * 10);
        println!("[Monitor] - Throughput: {} records/sec (estimated)", 10000 + iteration * 100);
        
        if iteration >= 20 {
            break;
        }
    }
    
    println!("Monitor stopped");
    Ok(())
}
```

## 🎯 Documented Project

### Description
Build a real-time fraud detection ETL pipeline that processes millions of financial transactions per day, applies machine learning models for anomaly detection, and triggers alerts for suspicious activity. The system uses Rust's async capabilities for high throughput and Polars for vectorized analytics.

### Functional Requirements
1. **Process 10M+ transactions/day** with sub-second latency per transaction
2. **Apply real-time ML scoring** using pre-trained XGBoost models
3. **Detect anomalies** using statistical methods (Z-score, IQR) and ML predictions
4. **Generate alerts** within 100ms of suspicious transaction detection
5. **Maintain exactly-once processing** with checkpointing and idempotent writes

### Main Components
- **Transaction Ingestor**: Kafka consumer with async message processing
- **Feature Engineer**: Real-time computation of transaction features (velocity, amount patterns)
- **ML Scorer**: ONNX runtime integration for model inference
- **Anomaly Detector**: Statistical and ML-based anomaly detection
- **Alert Generator**: Priority queue for alert management and notification
- **Checkpoint Manager**: State management for exactly-once processing

### Success Metrics
- **Throughput**: > 10,000 transactions/second sustained
- **Latency**: < 100ms P99 from ingestion to alert generation
- **Accuracy**: > 95% precision, > 90% recall on fraud detection
- **Availability**: 99.9% uptime with automatic failover
- **Cost**: < $0.01 per 1000 transactions processed

### References
- [Tokio Documentation](https://tokio.rs/)
- [Async Book](https://rust-lang.github.io/async-book/)
- [DataFusion Query Engine](https://arrow.apache.org/datafusion/)
- [Polars User Guide](https://pola.rs/polars/py-lazy/polars.html)
- [Rust ETL Patterns](https://www.shuttle.rs/blog/2023/09/27/rust-etl-pipelines)