https://kgptalkie.com/viva-question-answer-on-cache-memory

# Viva Question Answer on Cache Memory

**Source:** https://kgptalkie.com/viva-question-answer-on-cache-memory

**Source:** https://kgptalkie.com/viva-question-answer-on-cache-memory

**Source:** https://kgptalkie.com/viva-question-answer-on-cache-memory

## Published on  
10 September 2016

### Q.1 What are the differences among sequential access, direct access, and random access?

**Ans:**  
- **Sequential access:**  
  Memory is organized into units of data, called records. Access must be made in a specific linear sequence.  
- **Direct access:**  
  Individual blocks or records have a unique address based on physical location. Access is accomplished by direct access to reach a general vicinity plus sequential searching, counting, or waiting to reach the final location.  
- **Random access:**  
  Each addressable location in memory has a unique, physically wired-in addressing mechanism. The time to access a given location is independent of the sequence of prior accesses and is constant.

### Q.2 What is the general relationship among access time, memory cost, and capacity?

**Ans:**  
- Faster access time, greater cost per bit.  
- Greater capacity, smaller cost per bit.  
- Greater capacity, slower access time.

### Q.3 How does the principle of locality relate to the use of multiple memory levels?

**Ans:**  
It is possible to organize data across a memory hierarchy such that the percentage of accesses to each successively lower level is substantially less than that of the level above. Because memory references tend to cluster, the data in the higher level memory need not change very often to satisfy memory access requests.

### Q.4 What is access time or latency in memory?

**Ans:**  
*Access time (latency):* For random-access memory, this is the time it takes to perform a read or write operation, that is, the time from the instant that an address is presented to the memory to the instant that data have been stored or made available for use. For non-random-access memory, access time is the time it takes to position the read–write mechanism at the desired location.

### Q.5 What is Transfer Rate?

**Ans:**  
This is the rate at which data can be transferred into or out of a memory unit. For random-access memory, it is equal to 1/(cycle time).

### Q.6 What is **cache memory**?

**Ans:**  
Computer memory is organized into a hierarchy. At the highest level (closest to the processor) are the processor registers. Next comes one or more levels of cache, when multiple levels are used, they are denoted L1, L2, and so on. It is fastest memory available to the processor. The cache contain copy of main memory, when processor try to access memory it first try to access from cache.

### Q.7 What are the differences among **direct mapping**, **associative mapping**, and **set associative mapping**?

**Ans:**  
In a cache system,  
- **Direct mapping** maps each block of main memory into only one possible cache line.  
- **Associative mapping** permits each main memory block to be loaded into any line of the cache.  
- **Set-associative mapping** divides the cache into a number of sets of cache lines; each main memory block can be mapped into any line in a particular set.

### Q.8 For a direct-mapped cache, a main memory address is viewed as consisting of three fields. List and define the three fields.

**Ans:**  
- One field identifies a unique word or byte within a block of main memory.  
- The remaining two fields specify one of the blocks of main memory. These two fields are:  
  - A **line field**, which identifies one of the lines of the cache.  
  - A **tag field**, which identifies one of the blocks that can fit into that line.

### Q.9 For an associative cache, a main memory address is viewed as consisting of two fields. List and define the two fields.

**Ans:**  
- A **tag field** uniquely identifies a block of main memory.  
- A **word field** identifies a unique word or byte within a block of main memory.

### Q.10 For a set-associative cache, a main memory address is viewed as consisting of three fields. List and define the three fields.

**Ans:**  
- One field identifies a unique word or byte within a block of main memory.  
- The remaining two fields specify one of the blocks of main memory. These two fields are:  
  - A **set field**, which identifies one of the sets of the cache.  
  - A **tag field**, which identifies one of the blocks that can fit into that set.

### Q.11 What is the distinction between **spatial locality** and **temporal locality**?

**Ans:**  
- **Spatial locality** refers to the tendency of execution to involve a number of memory locations that are clustered.  
- **Temporal locality** refers to the tendency for a processor to access memory locations that have been used recently.

### Q.12 In general, what are the strategies for exploiting spatial locality and temporal locality?

**Ans:**  
- **Spatial locality** is generally exploited by using larger cache blocks and by incorporating prefetching mechanisms (fetching items of anticipated use) into the cache control logic.  
- **Temporal locality** is exploited by keeping recently used instruction and data values in cache memory and by exploiting a cache hierarchy.