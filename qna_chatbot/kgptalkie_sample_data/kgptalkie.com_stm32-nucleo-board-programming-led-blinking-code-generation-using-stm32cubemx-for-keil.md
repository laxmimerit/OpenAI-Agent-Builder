[https://kgptalkie.com/stm32-nucleo-board-programming-led-blinking-code-generation-using-stm32cubemx-for-keil](https://kgptalkie.com/stm32-nucleo-board-programming-led-blinking-code-generation-using-stm32cubemx-for-keil)

# STM32 Nucleo Board Programming – LED Blinking Code Generation Using STM32CubeMx for Keil

**Source:** [https://kgptalkie.com/stm32-nucleo-board-programming-led-blinking-code-generation-using-stm32cubemx-for-keil](https://kgptalkie.com/stm32-nucleo-board-programming-led-blinking-code-generation-using-stm32cubemx-for-keil)

**Source:** [https://kgptalkie.com/stm32-nucleo-board-programming-led-blinking-code-generation-using-stm32cubemx-for-keil](https://kgptalkie.com/stm32-nucleo-board-programming-led-blinking-code-generation-using-stm32cubemx-for-keil)

**Source:** [https://kgptalkie.com/stm32-nucleo-board-programming-led-blinking-code-generation-using-stm32cubemx-for-keil](https://kgptalkie.com/stm32-nucleo-board-programming-led-blinking-code-generation-using-stm32cubemx-for-keil)

## Published Date

10 September 2016  
10 September 2016

---

### Introduction

Today we present the first steps with the **NUCLEO development boards**, produced by **STMicroelectronics**, that can help us move towards the **ARM 32-bit world** with simplicity and great performances, keeping a compatibility with **Arduino expansion connectors** so that we can use its commonly available shields.

The success of **Arduino** and its countless shields has kicked off in recent years the birth of several compatible development boards designed to help us create in a short time, at low cost, and easily, great and even complex electronic applications. Some of these boards are simple clones, others are at much higher level having better performances and memory storage.

Among those, a really interesting solution is represented by the **NUCLEO development boards family** made by **STMicroelectronics**, a semiconductors leader company.

---

### NUCLEO F401RE Board Overview

In this post, we will examine the **NUCLEO F401RE board**, which is among the best performing in the series, not only because it is based on an **ARM processor** with a **84 MHz clock**, a **512 Kb flash memory**, and an **integrated floating-point unit**, but also for a number of interesting features that we will see together.

The board name comes from the **microcontroller mounted on the board (STM32F401)**, which is its heart. The whole series of **NUCLEO development boards** is equipped with a **STM32 microcontroller** based on **ARM Cortex-M family**, adopting a **32-bit RISC architecture**. Each **NUCLEO board** differs for performances, power consumption, clock frequency, and flash memory capacity of the **STM32 microcontroller**.

All the boards, however, have the same layout and the same form, which is shown in the next figure.

---

### Board Features

From here on, we will analyze the **NUCLEO model F401** and we will move our first programming steps, but many of the aspects and features that we will see later will be valid for any other **NUCLEO board**.

#### Connector Layout

One of the first aspects that we can note is the presence of many contacts on the card’s border, including the now famous **female contacts connector** compatible with the **Arduino shield**. Externally, however, two double strips of **male contacts** (one per side) are what **STM calls “Morpho” pinout**, used on other **STM development boards**.

- **Arduino pinout** is shown in **purple**, while the **Morpho pinout** is in **blue**.
- Notice how all **Arduino pins** are remapped exactly on **Morpho inner pin strip** (connectors **CN7** and **CN10**): this allows us to always have access to **Arduino pinout** even once a shield is plugged on the board. This helps us to debug software easily and to use those outputs when some shields don’t pass-through.
- **CN7** and **CN10** connectors pins are not connected to **Arduino compatible connector** and they provide other proprietary **I/O or power connector** typical of **STM32 microcontrollers**. This allows the card to be used in other projects which require greater connectivity.
- **CN7** and **CN10** **Morpho connectors** are replicated also on the board backside (always with **male contacts strips**), allowing you to mount the **NUCLEO board** on another board that could be seen as a new shield and that can access (also and not only) to **Arduino pinout**.

---

### Debugging and Programming

Another interesting feature is the presence on the **NUCLEO board** of a **PCB area** that is always part of the board, but serves exclusively to its programming and debugging. It is the **PCB part**, looking in the figure, that is close to the two small buttons and that can easily be physically split; this helps reducing the **NUCLEO board size** that actually runs the applications.

Specifically, it is the **ST-LINK/V2 debugger** (further details in the box on these pages) manufactured by **STMicroelectronics** that is in our case integrated on the same board, without needing additional hardware (and costs). In fact, the same **USB cable**, which is used to power up will also serve to program and debug our **NUCLEO board**, as we shall see later.

Once you’ve programmed your board, you can tear-off the debugger board and have in this way a very compact microcontroller board. It will always be possible to program and debug the **NUCLEO board** again, by connecting with external cables, the **SWD connector (CN4)** on the debug board to **Morpho connector (CN7)** pins **15 and 17** on the **NUCLEO board**.

The **SWD (Serial Wire Debug)** protocol recently introduced by **ARM** and implemented in all **Cortex-M microcontroller family** is transported, in fact, over only two wires instead of the five-wire **JTAG** that we are usually accustomed to.

So, unless special needs and at least in these early stages of development, we do not recommend to separate the two boards, because having all integrated is much more comfortable for our purpose (taking first steps with our **STM32 system**) and also we will have a unique power that is supplied from the **USB cable** through **CN1**.

This portion of the circuit is independent from the rest and is always equipped with a **STM32 microcontroller** suitably programmed during manufacture to manage the functions of a real programmer and debugger for the **STM8 and STM32 family of microcontrollers**.

**Source:** [https://kgptalkie.com/stm32-nucleo-board-programming-led-blinking-code-generation-using-stm32cubemx-for-keil](https://kgptalkie.com/stm32-nucleo-board-programming-led-blinking-code-generation-using-stm32cubemx-for-keil)