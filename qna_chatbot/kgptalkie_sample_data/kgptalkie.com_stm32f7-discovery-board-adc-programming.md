[https://kgptalkie.com/stm32f7-discovery-board-adc-programming](https://kgptalkie.com/stm32f7-discovery-board-adc-programming)

# STM32F7 Discovery Board ADC Programming

Published by  
on  
10 September 2016  
10 September 2016  

The STM32F7 Discovery Board features **3 ADCs** on board, each being a **12-bit ADC**. The 12-bit ADC is a successive approximation analog-to-digital converter. It supports up to **19 multiplexed channels**, allowing measurement of signals from **16 external sources**, **two internal sources**, and the **VBAT channel**. A/D conversions can be performed in **single**, **continuous**, **scan**, or **discontinuous mode**. The ADC result is stored in a **left or right-aligned 16-bit data register**.

The **analog watchdog** feature detects if input voltage exceeds user-defined **higher or lower thresholds**.

## ADC Main Features

- **12-bit, 10-bit, 8-bit, or 6-bit configurable resolution**
- **Interrupt generation** for:
  - End of conversion
  - End of injected conversion
  - Analog watchdog or overrun events
- **Single and continuous conversion modes**
- **Scan mode** for automatic conversion of channels 0 to n
- **Data alignment** with in-built data coherency
- **Channel-wise programmable sampling time**
- **External trigger option** with configurable polarity for regular and injected conversions
- **Discontinuous mode**
- **Dual/Triple mode** (on devices with 2 ADCs or more)
- **Configurable DMA data storage** in Dual/Triple ADC mode
- **Configurable delay** between conversions in Dual/Triple interleaved mode
- **ADC supply requirements**: 2.4 V to 3.6 V at full speed, down to 1.8 V at slower speed
- **ADC input range**: VREF– ≤ VIN ≤ VREF+
- **DMA request generation** during regular channel conversion

For more details, refer to: [https://kgptalkie.com/stm32f7-discovery-board-adc-programming](https://kgptalkie.com/stm32f7-discovery-board-adc-programming)

## Channel Selection

There are **16 multiplexed channels**. Conversions can be organized into two groups: **regular** and **injected**. A group consists of a sequence of conversions on any channel in any order. For example:

- ADC_IN3
- ADC_IN8
- ADC_IN2
- ADC_IN2
- ADC_IN0
- ADC_IN2
- ADC_IN2
- ADC_IN15

ADC1, ADC2, and ADC3 are tightly coupled and share some external channels. For more details, refer to: [https://kgptalkie.com/stm32f7-discovery-board-adc-programming](https://kgptalkie.com/stm32f7-discovery-board-adc-programming)