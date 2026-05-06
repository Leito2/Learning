# 📟 Embedded Rust and IoT

## Introduction

Rust is gaining traction in embedded systems due to its zero‑cost abstractions, memory safety, and fearless concurrency. The `no_std` attribute allows Rust to run without the standard library, making it suitable for microcontrollers with limited RAM and flash. The Embedded Rust Working Group has established a hardware abstraction layer (HAL) ecosystem based on traits, enabling portable driver code across different chips.

Embedded Rust leverages the same ownership and borrowing system that prevents data races in application software, but now at the hardware register level. This prevents common bugs like accidental register corruption, concurrent access to peripherals without synchronization, and buffer overflows in communication buffers.

This deep dive explores the `no_std` environment, the `embedded‑hal` traits, target architectures (ARM Cortex‑M, RISC‑V, ESP32), and practical examples like blinking an LED on an STM32 microcontroller. We’ll also examine real‑world applications, such as drone autopilots written in Rust.

## 1. no_std: Rust Without the Standard Library

`no_std` is a crate‑level attribute that removes the standard library dependency. The `core` crate provides essential types (`Option`, `Result`, `Vec`‑like collections via `alloc`), and `alloc` can be optionally imported for heap allocation if a global allocator is present.

### Key Differences

- **No threads**: Embedded systems are typically single‑threaded or use cooperative scheduling.
- **No file I/O**: No filesystem; I/O is via peripheral registers or hardware interfaces.
- **No dynamic linking**: Everything is statically linked.
- **No runtime**: No panic handling by default; you must define a panic handler.

**Real case:** The RTIC (Real‑Time Interrupt‑driven Concurrency) framework uses `no_std` and provides a task‑based scheduler with guaranteed worst‑case execution times.

⚠️ **Warning:** Using `std` features (e.g., `String`, `Vec`) in `no_std` requires a global allocator and the `alloc` crate. Many microcontrollers lack sufficient RAM for dynamic allocation.

💡 **Tip:** Prefer static allocation (arrays, `static mut`) over dynamic allocation in embedded systems. Use `heapless` crate for fixed‑capacity collections.

## 2. HAL (Hardware Abstraction Layer)

The `embedded‑hal` crate defines traits for common hardware interfaces (GPIO, SPI, I²C, UART, timers). Implementations are provided by chip‑specific crates (e.g., `stm32f4xx‑hal`), allowing drivers to be written once and used across different MCUs.

### Key Traits

| Trait | Description | Example Usage |
|-------|-------------|---------------|
| `OutputPin` | Digital output (set high/low) | LED control |
| `InputPin` | Digital input (read state) | Button reading |
| `SpiMaster` | SPI communication | Sensors, displays |
| `I2cMaster` | I²C communication | Accelerometers, EEPROMs |
| `Uart` | Serial communication | Debug console, GPS modules |
| `Timer` | Delay and PWM generation | Servo control, tone generation |

**Real case:** The `probe‑rs` debugging tool uses `embedded‑hal` traits to support multiple probe types without code duplication.

⚠️ **Warning:** Traits may be asynchronous (e.g., `Future`‑based) or blocking. Ensure your driver matches the HAL's concurrency model.

💡 **Tip:** Use `nb` (non‑blocking) crate for interrupt‑driven I/O without requiring async/await.

## 3. Targets: ARM Cortex‑M, RISC‑V, ESP32

Rust supports several embedded targets via LLVM backends:

| Architecture | Typical MCU | Flash/RAM | Rust Target | Notes |
|--------------|-------------|-----------|-------------|-------|
| ARM Cortex‑M0/M0+ | STM32F0, nRF52810 | 64‑256 KB / 8‑32 KB | `thumbv6m‑none‑eabi` | Minimal instructions, no atomic ops |
| ARM Cortex‑M4/M7 | STM32F4, STM32H7 | 1‑2 MB / 256 KB‑1 MB | `thumbv7em‑none‑eabihf` | FPU, DSP extensions |
| RISC‑V RV32IMC | ESP32‑C3, ESP32‑C6 | 4‑16 MB / 400 KB‑8 MB | `riscv32i‑unknown‑none‑elf` | Growing ecosystem |
| ESP32‑Xtensa | ESP32, ESP32‑S2 | 4‑16 MB / 520 KB | `xtensa‑esp32‑none‑elf` | Proprietary, requires custom LLVM |

**Real case:** The Dronecode autopilot (for UAVs) uses Rust on Cortex‑M7 for flight control algorithms.

Formula: `Flash_Usage = Code + Data + Stack + Heap`. Estimate each component to ensure fits in flash.

⚠️ **Warning:** Atomic operations are not available on Cortex‑M0 (no `LDREX`/`STREX`). Use critical sections instead.

💡 **Tip:** Use `cargo‑embed` or `probe‑rs` for flashing and debugging; they support RTT (Real‑Time Transfer) for efficient logging.

## 4. Blinking LED on STM32

Here’s a minimal example of blinking an LED on an STM32F4 using `embedded‑hal` and `stm32f4xx‑hal`.

```rust
#![no_std]
#![no_main]

use panic_halt as _;
use stm32f4xx_hal::{pac, prelude::*};

#[cortex_m_rt::entry]
fn main() -> ! {
    let dp = pac::Peripherals::take().unwrap();
    let gpioa = dp.GPIOA.split();
    let mut led = gpioa.pa5.into_push_pull_output();

    let cp = cortex_m::Peripherals::take().unwrap();
    let mut syst = cp.SYST;

    syst.set_clock_source(cortex_m::peripheral::syst::SystClkSource::Core);
    syst.set_reload(8_000_000); // 1 second at 8 MHz
    syst.clear_current();
    syst.enable_counter();

    loop {
        led.set_high();
        while !syst.has_wrapped() {}
        led.set_low();
        while !syst.has_wrapped() {}
        syst.clear_current();
    }
}
```

---

## 📦 Compression Code

Complete Rust script for a temperature sensor logger using ADC, I²C, and UART.

```rust
// sensor_logger.rs
#![no_std]
#![no_main]

use cortex_m_rt::entry;
use panic_halt as _;
use stm32f4xx_hal::{adc::Adc, i2c::I2c, pac, prelude::*, serial::Serial, timer::Timer};
use embedded_hal::blocking::i2c::WriteRead;

struct TempSensor {
    address: u8,
}

impl TempSensor {
    fn new(address: u8) -> Self {
        Self { address }
    }

    fn read_temp(&self, i2c: &mut impl WriteRead) -> Result<f32, ()> {
        let mut buf = [0; 2];
        i2c.write_read(self.address, &[0x00], &mut buf).map_err(|_| ())?;
        let raw = ((buf[0] as u16) << 8) | (buf[1] as u16);
        let temp = (raw as f32) * 0.0625; // Example conversion
        Ok(temp)
    }
}

#[entry]
fn main() -> ! {
    let dp = pac::Peripherals::take().unwrap();
    let cp = cortex_m::Peripherals::take().unwrap();

    let rcc = dp.RCC.constrain();
    let clocks = rcc.cfgr.freeze(&mut dp.FLASH.constrain().acr);

    let gpioa = dp.GPIOA.split();
    let gpiob = dp.GPIOB.split();

    // UART for logging
    let tx = gpioa.pa2.into_alternate();
    let rx = gpioa.pa3.into_alternate();
    let serial = Serial::new(dp.USART2, (tx, rx), 115_200.bps(), &clocks).unwrap();
    let (mut tx, _rx) = serial.split();

    // I²C for temperature sensor (e.g., LM75)
    let scl = gpiob.pb6.into_alternate_open_drain();
    let sda = gpiob.pb7.into_alternate_open_drain();
    let i2c = I2c::new(dp.I2C1, (scl, sda), 100.khz(), &clocks);

    let sensor = TempSensor::new(0x48); // LM75 address
    let mut timer = Timer::tim2(dp.TIM2, &clocks);

    loop {
        if let Ok(temp) = sensor.read_temp(&mut i2c) {
            // Echo temperature over UART (simplified)
            let _ = tx.write(b"Temp: ");
            let mut buf = [0; 16];
            let s = format_float(temp, &mut buf);
            let _ = tx.write(s.as_bytes());
            let _ = tx.write(b"\r\n");
        }
        timer.start(1_000_000_us); // 1 second
        block!(timer.wait()).unwrap();
    }
}

fn format_float(value: f32, buf: &mut [u8]) -> &str {
    // Simplified formatting
    let int = value as i32;
    let frac = ((value - int as f32) * 100.0) as u32;
    let s = alloc::format!("{}.{:02}", int, frac);
    buf[..s.len()].copy_from_slice(s.as_bytes());
    core::str::from_utf8(&buf[..s.len()]).unwrap()
}
```

## 🎯 Documented Project

### Description

A battery‑powered environmental monitor that logs temperature, humidity, and air quality to an SD card, with Bluetooth Low Energy (BLE) reporting.

### Functional Requirements

1. Read data from I²C sensors (SHT30, SGP30).
2. Log to SD card every 5 minutes (FAT32 filesystem).
3. Broadcast data via BLE every 30 seconds.
4. Deep sleep between measurements to conserve battery.
5. Configurable thresholds with LED alerts.

### Main Components

- **Sensor Driver**: `embedded‑hal`‑based I²C drivers.
- **SD Card**: SPI with `embedded‑sdmmc` crate.
- **BLE**: `embassy`‑BLE stack on nRF52832.
- **Power Management**: `nrf‑power` driver for deep sleep.
- **RTC**: Real‑time clock for timestamping.

### Success Metrics

- Battery life > 6 months on CR2032.
- Data loss < 0.1% after 30‑day test.
- BLE range > 10 meters line‑of‑sight.
- Cold start boot time < 2 seconds.

### References

- Embedded Rust book: [[04 - Embedded Rust and IoT|embedded‑hal]]
- RTIC framework documentation
- probe‑rs debugging guide
- STM32F4 reference manual