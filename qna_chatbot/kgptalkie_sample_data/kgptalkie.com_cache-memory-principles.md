**Source:** https://kgptalkie.com/cache-memory-principles

# Cache Memory Principles

**Source:** https://kgptalkie.com/cache-memory-principles

## Published by

13 September 2016

---

### Overview

Cache memory is intended to give memory speed approaching that of the fastest memories available, and at the same time provide a large memory size at the price of less expensive types of semiconductor memories. The concept is illustrated in **Figure (a)** below.

There is a relatively large and slow main memory together with a smaller, faster cache memory. The cache contains a copy of portions of main memory.

When the processor attempts to read a word of memory, a check is made to determine if the word is in the cache. If so, the word is delivered to the processor. If not, a block of main memory, consisting of some fixed number of words, is read into the cache and then the word is delivered to the processor. Because of the phenomenon of **locality of reference**, when a block of data is fetched into the cache to satisfy a single memory reference, it is likely that there will be future references to that same memory location or to other words in the block.

---

### Multi-Level Cache

**Figure (b)** depicts the use of multiple levels of cache. The L2 cache is slower and typically larger than the L1 cache, and the L3 cache is slower and typically larger than the L2 cache.

---

### Cache/Main Memory System Structure

**Source:** https://kgptalkie.com/cache-memory-principles

Main memory consists of up to $2^n$ addressable words, with each word having a unique $n$-bit address. For mapping purposes, this memory is considered to consist of a number of fixed-length blocks of $K$ words each. That is, there are $M = \frac{(2^n)}{K}$ blocks in main memory.

The cache consists of $m$ blocks, called **lines**. Each line contains $K$ words, plus a tag of a few bits. Each line also includes control bits (not shown), such as a bit to indicate whether the line has been modified since being loaded into the cache.

The length of a line, not including tag and control bits, is the **line size**. The line size may be as small as 32 bits, with each “word” being a single byte; in this case, the line size is 4 bytes.

The number of lines is considerably less than the number of main memory blocks ($m << M$). At any time, some subset of the blocks of memory resides in lines in the cache. If a word in a block of memory is read, that block is transferred to one of the lines of the cache. Because there are more blocks than lines, an individual line cannot be uniquely and permanently dedicated to a particular block. Thus, each line includes a **tag** that identifies which particular block is currently being stored. The tag is usually a portion of the main memory address, as described later in this section.

---

### Read Operation

**Figure (a)** illustrates the read operation. The processor generates the read address (RA) of a word to be read. If the word is contained in the cache, it is delivered to the processor. Otherwise, the block containing that word is loaded into the cache, and the word is delivered to the processor. **Figure (b)** shows these last two operations occurring in parallel and reflects the organization shown in **Figure (c)**, which is typical of contemporary cache organizations.

In this organization, the cache connects to the processor via data, control, and address lines. The data and address lines also attach to data and address buffers, which attach to a system bus from which main memory is reached.

When a cache hit occurs, the data and address buffers are disabled and communication is only between processor and cache, with no system bus traffic. When a cache miss occurs, the desired address is loaded onto the system bus and the data are returned through the data buffer to both the cache and the processor.

In other organizations, the cache is physically interposed between the processor and the main memory for all data, address, and control lines. In this latter case, for a cache miss, the desired word is first read into the cache and then transferred from cache to processor.

**Source:** https://kgptalkie.com/cache-memory-principles