[https://kgptalkie.com/a-guide-to-sub-1-ghz-long-range-communication-and-smartphone-connection-for-iot-application-to-lowpower-rf-connectivity](https://kgptalkie.com/a-guide-to-sub-1-ghz-long-range-communication-and-smartphone-connection-for-iot-application-to-lowpower-rf-connectivity)

# A Guide to Sub-1 GHz Long-Range Communication and Smartphone Connection for IoT Application to Lowpower RF Connectivity

**Source:** [https://kgptalkie.com/a-guide-to-sub-1-ghz-long-range-communication-and-smartphone-connection-for-iot-application-to-lowpower-rf-connectivity](https://kgptalkie.com/a-guide-to-sub-1-ghz-long-range-communication-and-smartphone-connection-for-iot-application-to-lowpower-rf-connectivity)

Published by  
on  
19 October 2016  
19 October 2016  

## Introduction

In today’s Internet of Things (IoT) world, there is a multitude of new wireless connectivity applications entering the market each day, propelling the continuous gathering of sensors and interactions. From our smart phone telling us how many steps we have taken to our security system telling us that no windows are left open, we have a safety net of reminders helping us effortlessly move throughout our day. This trend of gathering more information creates daily interactions with different wireless devices. Within one day a person will interface with over 100 connected things using multiple wireless protocols or standards. As of now, there is very little overlap as you connect from your home security system to your car to your office. The interface is a bit awkward as you switch from wireless bands and separate networks, so how do you encourage more interaction between these networks? What is often missing is the seamless interaction from 2.4 GHz to Sub-1 GHz.

## Sub-1 GHz: Long-Range and Low Power RF Connectivity

For a lot of wireless products, the range is much more important than being able to send high throughput data. Take smart metering, for example, or a sensor device in an alarm system, or a temperature sensor in a home automation system. For these applications, the Sub-1 GHz industrial scientific and medical (ISM) bands (433/868/915 MHz) offer much better range than a solution using the 2.4GHz band. The main reason for this is the physical property of the lower frequency. Given the same antenna performance, this theory (free space) calls for twice the range when using half the RF frequency. Another important factor is that the longer RF waves have an ability to pass through walls and bend around corners. The lower data rate will also play a part since the sensitivity of the receiver is a strong function of the data rate. As a rule of thumb, a reduction of the data rate by a factor of four will double the range (free space). Lastly, due to the low-duty cycle allowed in the Sub-1 GHz RF regulations, there are fewer issues with disturbances for low-data-rate solutions in the Sub-1 GHz bands than the 2.4-GHz band (mainly due to Wi-Fi®).

The lower frequency also helps to keep the current consumption low. In addition to offering higher battery life, the lower peak current consumption also enables a smaller form factor solution using coin cell batteries. However, getting the data from the Sub-1 GHz system into your smart device can be challenging, mostly due to the fact that smart devices do not typically include Sub-1 GHz communication systems for use with ISM band communication. For this reason, **Bluetooth® low energy** is the de-facto standard to use, which is where a dual-band wireless microcontroller (MCU) can act as a bridge between the two communication bands. With the **SimpleLink™ dual-band CC1350 wireless MCU**, combining Sub-1 GHz and Bluetooth low energy is now possible. The CC1350 device is able to transmit +10 dBm using only 15mA, which is perfectly okay to handle for a coin cell battery. Using low data rates—it is possible to transmit over 20 km (line of sight from an elevated transmitter) with the RF receiver consumption being only 5.4 mA using a 3.6-V lithium battery.

## Challenges with the Sub-1 GHz Bands

It is easy to appreciate the range and low power using the Sub-1 GHz band, but naturally, there are also some drawbacks. As described earlier, one of the main tools used in our daily life, the smartphone, does not use Sub-1 GHz. Or actually, it does, it is using the licensed bands (GPRS, 3G, and LTE) to get the best range, but it is not using the Sub-1 GHz ISM bands. The fact that both Wi-Fi and Bluetooth are standard features of any smartphone available on the market today offers a clear advantage for those technologies. An obvious solution to this is to combine the best of two worlds—Sub-1 GHz technology for long range and low power and a 2.4-GHz solution using Bluetooth low energy for a smartphone/tablet/PC connection. The first RF IC publicly available on the market that can do this is the **CC1350 wireless MCU** from Texas Instruments (TI). The CC1350 device is a single-chip solution that includes a high-efficiency ARM® Cortex®-M3 MCU, a low-power sensor controller, and a low-power dual-band RF transceiver.

### SimpleLink Dual-Band CC1350 Wireless MCU

- **ARM Cortex-M3 application processor**: 128 kB Flash, 20 kB ultra-low power SRAM, 8 kB SRAM for cache (can also be allocated as regular SRAM)
- **RF core**: RF front-end capable of supporting the most relevant Sub-1 GHz bands (315, 433, 470, 868, 915 MHz) as well as 2.4 GHz
- **Radio core**: Very flexible software-configurable modem to cover data rates from a few hundred bits per second up to 4 Mbps and multiple modulation formats from “simple” OOK (on–off keying), to (G)FSK, (G)MSK, 4-(G)FSK and shaped 8-FSK
- **ARM Cortex-M0 in the RF core**: Running pre-programmed ROM functions to support both low-level Bluetooth and proprietary RF solutions

The power system tightly integrates a digital converter to digital converter (DC/DC) solution that is active in all modes of operation, including standby. This ensures low-power operation, as well as stable performance (RF range) despite the drop in battery voltage.

### ROM in CC1350 Wireless MCU

The **SimpleLink CC1350** device contains over 200kB of ROM (Read Only Memory) with libraries covering the following functions:

- TI-RTOS (real time operating system)
- Low-level driver library (SPI, UART, etc.)
- Security functions
- Low level and some higher level, Bluetooth stack functions

Note that ROM code can be fixed/patched by functions in Flash or RAM.

## Ultra-Low Current Consumption

The **SimpleLink CC1350** and **CC1310** (Sub-1 GHz only) wireless MCUs offer ultra-low current consumption in all modes of the operation both for the RF as well as the microcontroller.

### The Sensor Controller

The sensor controller is a native, small power-optimized 16-bit MCU that is included in the **CC13xx** devices to handle analog and digital sensors in a very low-power manner. It is programmed/configured using the **Sensor Controller Studio** where users find predefined functions for the different peripherals. The tool also offers software examples of common sensor solutions like ADC reading (streaming, logging window compare functions) and I²C/SPI for digital sensors. The sensor controller can also be used for capacitive touch buttons.

## Bluetooth Low Energy Software Stacks

**Bluetooth** low energy software stacks: TI offered one of the first certified **Bluetooth** low energy software stacks. The stack has since been developed further to support the **SimpleLink CC26xx** platform that was released in 2015. This stack is now also available for the **CC1350** device and has all the features that the **Bluetooth 4.2** standard offers—from “simple” beacons to a fully connectable stack. All TI RF stacks are using **TI RTOS**, a free real-time operating system from TI. **TI-RTOS** is distributed under the 3-Clause BSD license, meaning that full source code is provided.

To further reduce the complexity of developing applications and let customers solely focus on their application development, TI provides a large set of peripheral drivers, including a performance optimized RF driver. The **TI-RTOS** for **CC13xx** and **CC26xx** software development kits (SDK) offers a large set of getting started examples. The RF examples serve as a great starting point for developing proprietary systems, all software examples are provided for the purpose of showing a performance-optimized usage of the various drivers.

For new product development, without the need to adhere to legacy products, a great solution is to use the new **TI 15.4-Stack** offering. **TI 15.4-Stack** is TI’s implementation of the **IEEE 802.15.4g/e** standards, enabling start type networks. It is offered (free of charge) in two versions:

- **Version optimized for European RF regulations (ETSI)**—using frequency agility and LBT (Listen before talk)
- **Version optimized for US RF regulations (FCC)**—using frequency hopping to enable highest output power

## Sub-1 GHz and Bluetooth Low Energy Use Cases

The fact that the **CC1350 wireless MCU** enables both Sub-1 GHz and Bluetooth low energy in a single device opens up a lot of possibilities. 

**Source:** [https://kgptalkie.com/a-guide-to-sub-1-ghz-long-range-communication-and-smartphone-connection-for-iot-application-to-lowpower-rf-connectivity](https://kgptalkie.com/a-guide-to-sub-1-ghz-long-range-communication-and-smartphone-connection-for-iot-application-to-lowpower-rf-connectivity)