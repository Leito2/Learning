# 📱 TinyML and Edge ML

## Introduction

TinyML runs machine learning models on microcontrollers and edge devices with milliwatt power budgets and kilobytes of RAM — think wake-word detection on smart speakers, predictive maintenance on factory sensors, and gesture recognition on earbuds. While cloud ML processes data in datacenters with teraflop GPUs, TinyML processes data on-device with ARM Cortex-M processors and specialized NPUs, enabling privacy-preserving, always-on, low-latency inference.

For AI engineers, TinyML represents the frontier where ML meets embedded systems — a skill that differentiates you in IoT, wearables, and industrial automation.

---

## The TinyML Stack

```
┌──────────────────────────────────────────────────────────────┐
│                    TINYML DEPLOYMENT STACK                    │
│                                                              │
│  Training (Cloud/Desktop)              Inference (Edge)      │
│  ┌─────────────────────┐             ┌──────────────────┐  │
│  │ PyTorch / TensorFlow │──Export──▶│ TFLite / ONNX     │  │
│  └─────────────────────┘             │ + Quantization    │  │
│                                      └────────┬─────────┘  │
│                                               │              │
│                                               ▼              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Runtime (MCU)                            │  │
│  │  ┌────────────┐  ┌──────────────┐  ┌──────────────┐ │  │
│  │  │ TFLite     │  │ ONNX Runtime │  │ CMSIS-NN     │ │  │
│  │  │ Micro      │  │ (Micro)      │  │ (ARM-opt)    │ │  │
│  │  └────────────┘  └──────────────┘  └──────────────┘ │  │
│  └──────────────────────────────────────────────────────┘  │
│                                               │              │
│                                               ▼              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Hardware                                 │  │
│  │  ARM Cortex-M4/M7 │ ESP32 │ Arduino │ Coral TPU     │  │
│  │  Memory: 256KB RAM │ 1MB Flash │ Clock: 80-480MHz   │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Techniques

### Quantization

Reduce model size by converting 32-bit floats to 8-bit integers:

| Type | Size Reduction | Accuracy Loss | When |
|---|---|---|---|
| **FP32 → FP16** | 2x | Negligible | Nvidia Jetson, newer MCUs |
| **FP32 → INT8** | 4x | 0.5-2% | Standard TinyML deployment |
| **INT8 → Binary/INT4** | 8-32x | 3-10% | Extreme memory constraints |

### Knowledge Distillation

Train a small "student" model to mimic a large "teacher" model — combining the accuracy of a large model with the size of a small model.

### Model Architecture Choices

| Architecture | Parameters | RAM Needed | Use Case |
|---|---|---|---|
| **MobileNetV1** | 4.2M | ~5MB | Image classification on edge |
| **TinyBERT** | 14.5M | ~50MB | NLP on mobile (too large for MCU) |
| **MicroNet** | <100K | <500KB | Keyword spotting, gesture recognition |
| **Custom CNN (3-5 layers)** | 20K-100K | 100-500KB | Custom sensor ML |

---

## Frameworks

| Framework | Target | Key Feature |
|---|---|---|
| **TensorFlow Lite Micro** | ARM Cortex-M, ESP32 | Most widely supported |
| **ONNX Runtime Mobile** | Android, iOS | Cross-platform ONNX models |
| **Edge Impulse** | No-code platform for MCUs | Rapid prototyping, sensor DSP |
| **MicroTVM (Apache TVM)** | Custom MCUs | Auto-tuning compiler for any target |
| **CoreML** (Apple) | iPhone, iPad, Apple Watch | Native Apple Neural Engine acceleration |

---

## ⚠️ Constraints

- **Memory:** A typical MCU has 256KB RAM. An uncompressed MobileNet requires ~16MB. Quantization and architecture selection are mandatory.
- **No OS:** No Python interpreter, no file system, no garbage collection. All code runs on bare metal or RTOS.
- **Power:** Always-on devices run on coin cell batteries for months/years. Every milliwatt matters.

---

## References

- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [Edge Impulse](https://www.edgeimpulse.com/)
- [TinyML Book (Warden & Situnayake)](https://www.oreilly.com/library/view/tinyml/9781492052036/)
