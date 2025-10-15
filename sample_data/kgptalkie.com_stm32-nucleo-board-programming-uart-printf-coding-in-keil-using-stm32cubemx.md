https://kgptalkie.com/stm32-nucleo-board-programming-uart-printf-coding-in-keil-using-stm32cubemx

# STM32 Nucleo Board Programming – UART printf Coding in Keil using STM32CubeMx

Published by  
on  
11 September 2016  
11 September 2016  

## NUCLEO-F401RE – STM32 Nucleo-64 development board

The **STM32 Nucleo-64 development board** with **STM32F401RE MCU** supports **Arduino** and **ST morpho connectivity** – **STMicroelectronics**.

### Key Features of STM32 Nucleo Board

- Affordable and flexible for prototyping with any STM32 microcontroller line
- Supports various combinations of performance, power consumption, and features
- **Arduino™ connectivity** and **ST Morpho headers** for expanding functionality with specialized shields
- Integrated **ST-LINK/V2-1 debugger and programmer** (no separate probe required)
- Includes **STM32 comprehensive software HAL library**, packaged software examples, and access to **mbed online resources**

---

### USART2 Interface Configuration

The **USART2 interface** on **PA2** and **PA3** of the STM32 microcontroller can be connected to:

- **ST-LINK MCU**
- **ST morpho connector**
- **Arduino connector**

#### Solder Bridge Settings

| Configuration | Description |
|--------------|-------------|
| **Default** | USART2 communication between target MCU and ST-LINK MCU enabled (for Virtual Com Port in mbed) <br> **SB13 and SB14 ON**, **SB62 and SB63 OFF** |
| **Alternate** | Communication between target MCU (PA2/D1 or PA3/D0) and shield/extension board <br> **SB62 and SB63 ON**, **SB13 and SB14 OFF** |

> In such case, it is possible to connect another USART to ST-LINK MCU using **flying wires** between **ST morpho connector** and **CN3**.

---

For more details, visit the original article:  
https://kgptalkie.com/stm32-nucleo-board-programming-uart-printf-coding-in-keil-using-stm32cubemx