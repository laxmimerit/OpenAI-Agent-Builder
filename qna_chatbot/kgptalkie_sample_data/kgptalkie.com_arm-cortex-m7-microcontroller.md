[https://kgptalkie.com/arm-cortex-m7-microcontroller](https://kgptalkie.com/arm-cortex-m7-microcontroller)

# ARM Cortex-M7 Microcontroller

**Source:** [https://kgptalkie.com/arm-cortex-m7-microcontroller](https://kgptalkie.com/arm-cortex-m7-microcontroller)

**Source:** [https://kgptalkie.com/arm-cortex-m7-microcontroller](https://kgptalkie.com/arm-cortex-m7-microcontroller)

**Source:** [https://kgptalkie.com/arm-cortex-m7-microcontroller](https://kgptalkie.com/arm-cortex-m7-microcontroller)

Published by  
on  
10 September 2016  
10 September 2016  

[youtube](https://www.youtube.com/watch?v=aPOO_sw1HhQ?list=PLc2rvfiptPSR0bzPjEsg5zmj0jvYMZLbV)

## Introduction

ARM has introduced a new processor of the Cortex M series, and this time it is **Cortex M7**. The **ARM M7 processor** is the most recent and highest performance member of the energy-efficient Cortex-M processor family. ARM quotes: *“The versatility and new memory features of the Cortex-M7 enable more powerful, smarter and reliable microcontrollers that can be used across a multitude of embedded applications.”*

## Performance Focus

The primary focus of the **Cortex-M7** is **improved performance**. ARM’s goal was to elevate the M series performance to a level previously unseen, while maintaining the M series’ signature such as small die size and tiny power consumption as well as the excellent responsiveness and ease-of-use of the **ARMv7-M architecture**.

There are at least two reasons ARM focused on performance for the **M7 processor**:

1. To further drive a wedge between traditional 8- and 16-bit microcontrollers and provide ARM a further differentiated market position.
2. To support the **Internet of Things** and **wearable device markets**.

Focusing on enhanced **DSP capabilities**, the **Cortex M7** is more suited to **audio and visual sensor hub processing** than any previous M series design.

## Key Features

### DSP Power

- The **Cortex M7** has **twice the DSP power** of the **M4** by executing **twice as many instructions simultaneously**.
- It can operate at a **higher clock frequency** than the **M4**.
- Backed by the **Keil CMSIS DSP library**.
- Includes a **single and double precision FPU**.

### Floating Point Unit (FPU)

The optional **Floating Point Unit (FPU)** provides:

- **Automated stacking** of floating-point state is delayed until the **ISR** attempts to execute a floating-point instruction.
- **Instructions for single-precision** data-processing operations.
- **Optional instructions for double-precision** data-processing operations.
- **Combined multiply and accumulate instructions** for increased precision.
- **Hardware support** for conversion, addition, subtraction, multiplication with optional accumulate, division, and square-root.

### Nested Vectored Interrupt Controller (NVIC)

- **Closely integrated** with the core to achieve **low-latency interrupt processing**.
- **1 to 240 configurable external interrupts** (configured at implementation).
- **Configurable levels of interrupt priority** from **8 to 256** (configured at implementation).
- **Dynamic reprioritization** of interrupts.
- **Support for tail-chaining** and **late arrival of interrupts**.
- Enables **back-to-back interrupt processing** without the overhead of state saving and restoration between interrupts.

### Memory Protection Unit (MPU)

- Used to manage **CPU accesses to memory** to prevent one task from accidentally corrupting the memory or resources used by any other active task.
- **Memory area** is organized into **up to 8 protected areas**, which can be divided into **8 subareas**.
- **Protection area sizes** range from **32 bytes** to the **whole 4 gigabytes** of addressable memory.

### Tightly Coupled Memory (TCM)

- A technology used by ARM’s partners to extend the effective caching of a single **M7 processor**.
- Seen previously in **A and R series designs**.
- **Performance** similar to a cache, but contents are directly controlled by the developer.
- **Critical code and data** can be placed inside **TCM** for **deterministic access** with high performance in routines such as **interrupt service requests**.
- Supports **up to 16 MB** of tightly coupled memory.

### AHB-Lite Peripheral (AHBP) Interface

- Provides **low latency system peripherals** access.
- Supports **unaligned memory accesses**.
- Includes a **write buffer** for buffering of write data.
- Supports **exclusive access transfers** for multiprocessor systems.

## Pipeline Architecture

- Features a **six-stage, dual-issue superscalar pipeline** with **single- and double-precision floating point units**.
- Can execute **two instructions at a time**.
- **Cortex-M4** can execute **just one instruction at a time**.
- **M7** can run at a **higher clock frequency** than **M4**.
- Together, these features give an **average two-times uplift in DSP performance** for **M7** over **M4**.

## Applications

By doubling the performance, ARM calculates that appliances and gadgets using the **M7** can more quickly perform the complex mathematics required to:

- **Finely control motor movement** in robots.
- **Analyse microphone**, **touchscreen**, and **other sensors data**.