https://kgptalkie.com/avr-microcontroller-memory-architecture

# AVR Microcontroller Memory Architecture

Published by  
on  
19 October 2016  
19 October 2016  

## What is a memory map?

The memory map of a microcontroller is a diagram which gives the size, type, and layout of the memories that are available in the microcontroller. The information uses to construct the memory map is extracted from the datasheet of the microcontroller.

The ATMega8515 microcontroller contains three (3) blocks of memory: **Program Memory**, **EEPROM Memory**, and **Data Memory**.

### Data Memory Contains:
- 32 8-bits General Purpose
- 64 8-bits Input/Output Registers
- 512 8-bits SRAM space

### Program Memory Contains:
- 8K byte Flash Memory
- Organized as 4K-16bits space

### EEPROM Memory Contains:
- 512 8-bits EEPROM space

[Watch related video](https://www.youtube.com/watch?v=OG2LF6MFI2U&index=3&list=PLc2rvfiptPSQQkHvN7ofRi5Acnn2AMNJ0)

## Flash

Flash is nonvolatile memory, which means it persists when power is removed. Its purpose is to hold instructions that the microcontroller executes. The amount of flash can range from 512 bytes on an ATTiny to 384K on an ATxmega384A1. AVR microcontrollers can be thought of having 2 modes: **flash programming** and **flash executing mode**.

By modifying fuse settings (BOOTSZ0 & BOOTSZ1 on the ATmega168), some AVR microcontrollers allow you to reserve sections of flash for a **bootloader** and reserved application flash section. The bootloader allows the flash programming process to be controlled by a flash resident program. Some bootloader applications might include:

- Decrypt encrypted flash files to prevent reverse engineering
- Implement a self-destruct sequence triggered by a tamper sensor
- Allow the device to be programmed from a TFTP server

## RAM

RAM is a volatile memory that stores the runtime state of the program being executed. The amount of RAM can range from 32 bytes on an ATTiny28L to 32KB on an ATxmega384A1. In many AVR microcontrollers RAM is split into 4 subsections:

1. General purpose registers
2. I/O registers
3. Extended I/O registers
4. Internal RAM

AVR microcontrollers have RAM on-chip but some AVRs (e.g. ATMega128) can use external RAM modules to extend what is built into the microcontroller.

## EEPROM

EEPROM is nonvolatile memory which is used to store data. The most common use is to store configurable parameters. The amount of EEPROM can range from 32 bytes on an ATTiny to 4KB on an XMega.

[Watch related video](https://www.youtube.com/watch?v=KsA4SiAtLvk&list=PLc2rvfiptPSQQkHvN7ofRi5Acnn2AMNJ0&index=6)