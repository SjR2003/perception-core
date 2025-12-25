# perception core

**perception core** is a high-performance, real-time visual perception system designed for multi-stream video analytics.
It is optimized for **GPU inference (TensorRT)**, **batch processing**, and **multi-process pipelines**, supporting both **detection** and **tracking** at scale.

---

## Features

* **Dynamic batching** with timeout handling and dummy frame compensation
* **TensorRT-accelerated YOLO inference** (YOLOv5 /YOLOv8 / YOLO11)
* **Real-time object tracking** using BYTETracker
* **Main / Sub stream separation**
  * Main stream: Detection + Tracking
  * Sub streams: Detection only
* **ZeroMQ PUB/SUB architecture**
* **Low-latency, high-throughput pipeline**
* **Graceful handling of missing frames**
* **Production-oriented multiprocessing design**

---

## Architecture Overview

```
            +----------------+
            |  Video Sources |
            |      (ZMQ)     |
            +--------+-------+
                     |
                     v
             +------------------+
             |   ZMQ Receiver   |
             +--------+---------+
                      |
              Per-Stream Buffers
                      |
                      v
            +---------------------+
            | FrameBatchSampler   |
            | (RR + timeout +     |
            |  dummy frames)      |
            +----------+----------+
                       |
                       v
            +----------------------+
            | BatchDecodePipeline  |
            | (ThreadPool)         |
            +----------+-----------+
                       |
                       v
            +----------------------+
            | TensorRT YOLO Model  |
            | (Detection / Track) |
            +----------+-----------+
                       |
                       v
            +----------------------+
            |   ZMQ Publisher      |
            +----------------------+
```

---

## Project Structure

```
perception/
├── main.py
├── pipelines/
│   ├── stream_pipeline.py
│   └── pipline_process.py
├── preprocess/
│   ├── batch_scheduler.py
│   └── batch.py
├── models/
│   └── yolo/
│       ├── exporter.py
│       ├── loader.py
│       ├── trt_detector.py
│       ├── detector.py
│       └── tracker.py
├── network/
│   ├── zmq_handler.py
│   └── proxy.py
├── utils/
│   ├── logger.py
│   ├── latency_logger.py
│   └── utils.py
└── data/
	├── models/
    └── engine_models/
```

---

## Configuration

The system is fully configurable via YAML files:

* **network.yaml**

  * ZMQ endpoints
  * REST API endpoint (coming soon)
  * KAFKA endpoints (coming soon)
* **main_stream.yaml**

  * Model settings
  * Batch settings
  * Streams config
* **sub_stream.yaml**

  * Model settings
  * Batch settings
  * Streams config

---

##  Running perception core

```bash
python perception/main.py \
  --network_config configs/network.yaml \
  --main_stream_config configs/main_stream.yaml \
  --sub_stream_config configs/sub_stream.yaml
```

---

## Model Support

* YOLOv5 / YOLOv8 / YOLO11 (`.pt` → TensorRT `.engine`)
* FP16 inference
* Fixed batch TensorRT engines
* Automatic engine caching

---

## Concurrency Model

* **Multiprocessing**

  * Main stream process
  * Sub stream process
* **ThreadPool**

  * JPEG decode & preprocessing
* **GPU-bound inference**

  * Strict batch consistency enforced

---

## Dummy Frame Strategy (It will be removed soon!!!)

To avoid TensorRT input shape errors:

* Missing frames are **replaced with last valid frames**
* Marked via `is_dummy = True`
* Automatically ignored by tracker & publisher

This guarantees:

* Stable batch size
* No invalid GPU inputs
* No pipeline stalls

---

## Performance

Typical latency on GTX-class GPUs: `1660ti`

* Decode + preprocess: **12–30 ms**
* TensorRT inference: **30–60 ms**
* Publish: **< 1 ms**

(Depends on batch size & stream count)

---

## Current Status

* Core pipeline stable
* TensorRT + YOLO integration
* Tracking working

