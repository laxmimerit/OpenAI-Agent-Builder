# Viva Question Answer on DRAM & SRAM Memory  
**Source:** https://kgptalkie.com/viva-question-answer-on-dram-sram-memory  
**Source:** https://kgptalkie.com/viva-question-answer-on-dram-sram-memory  
**Source:** https://kgptalkie.com/viva-question-answer-on-dram-sram-memory  

## Published by  
on  
10 September 2016  
10 September 2016  

### Q.1 What are the key properties of semiconductor memory?  
**Ans:**  
- They exhibit two stable (or semistable) states, which can be used to represent binary **1** and **0**.  
- They are capable of being written into (at least once), to set the state.  
- They are capable of being read to sense the state.  

### Q.2 What are two senses in which the term random-access memory is used?  
**Ans:**  
1. A memory in which individual words of memory are directly accessed through wired-in addressing logic.  
2. Semiconductor main memory in which it is possible both to **read data** from the memory and to **write new data** into the memory easily and rapidly.  

### Q.3 What is the difference between DRAM and SRAM in terms of application?  
**Ans:**  
- **SRAM** is used for **cache memory** (both on and off chip).  
- **DRAM** is used for **main memory**.  

### Q.4 What is the difference between DRAM and SRAM in terms of characteristics such as speed, size, and cost?  
**Ans:**  
- **SRAMs** generally have **faster access times** than **DRAMs**.  
- **DRAMs** are **less expensive** and **smaller** than **SRAMs**.  

### Q.5 Explain why one type of RAM is considered to be analog and the other digital.  
**Ans:**  
- A **DRAM cell** is essentially an **analog device** using a **capacitor**; the capacitor can store any charge value within a range; a threshold value determines whether the charge is interpreted as **1** or **0**.  
- A **SRAM cell** is a **digital device**, in which binary values are stored using traditional **flip-flop logic-gate configurations**.  

### Q.6 What are some applications for ROM?  
**Ans:**  
- **Microprogrammed control unit memory**.  
- **Library subroutines** for frequently wanted functions.  
- **System programs**.  
- **Function tables**.  

### Q.7 What are the differences among EPROM, EEPROM, and flash memory?  
**Ans:**  
- **EPROM**:  
  - Read and written electrically.  
  - All storage cells must be erased to the same initial state by exposure to **ultraviolet radiation**.  
  - Erasure is performed by shining an **intense ultraviolet light** through a window in the memory chip.  
- **EEPROM**:  
  - Read mostly memory that can be written into at any time without erasing prior contents.  
  - Only the **byte or bytes addressed** are updated.  
- **Flash memory**:  
  - Intermediate between **EPROM** and **EEPROM** in cost and functionality.  
  - Uses **electrical erasing technology**.  
  - Entire flash memory can be erased in **one or a few seconds**.  
  - Supports **block-level erasure** (not entire chip).  
  - Does not provide **byte-level erasure**.  
  - Uses **one transistor per bit**, achieving high density compared to **EEPROM**.  

### Q.8 What is a parity bit?  
**Ans:**  
A bit appended to an array of binary digits to make the sum of all the binary digits, including the parity bit, always **odd** (**odd parity**) or always **even** (**even parity**).  

### Q.9 How is the syndrome for the Hamming code interpreted?  
**Ans:**  
- A **syndrome** is created by the **XOR** of the code in a word with a calculated version of that code.  
- Each bit of the syndrome is **0** or **1** according to whether there is or is not a match in that bit position for the two inputs.  
- If the syndrome contains **all 0s**, no error has been detected.  
- If the syndrome contains **one and only one bit set to 1**, an error has occurred in one of the **4 check bits** (no correction needed).  
- If the syndrome contains **more than one bit set to 1**, the **numerical value** of the syndrome indicates the position of the **data bit in error**. This data bit is **inverted** for correction.  

### Q.10 How does SDRAM differ from ordinary DRAM?  
**Ans:**  
- Unlike traditional **DRAM**, which is **asynchronous**, **SDRAM** exchanges data with the processor **synchronized to an external clock signal**.  
- **SDRAM** runs at the **full speed of the processor/memory bus** without imposing **wait states**.  

**Source:** https://kgptalkie.com/viva-question-answer-on-dram-sram-memory  
**Source:** https://kgptalkie.com/viva-question-answer-on-dram-sram-memory  
**Source:** https://kgptalkie.com/viva-question-answer-on-dram-sram-memory