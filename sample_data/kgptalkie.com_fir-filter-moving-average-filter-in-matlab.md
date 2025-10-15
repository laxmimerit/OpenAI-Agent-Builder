https://kgptalkie.com/fir-filter-moving-average-filter-in-matlab

# Moving Average FIR Filter in MATLAB

## Published by
on  
10 September 2016  
10 September 2016  

As the name implies, the moving average filter operates by averaging a number of points from the input signal to produce each point in the output signal. In equation form, this is written:

$$
y[n] = \frac{1}{M} \sum_{k=0}^{M-1} x[n-k]
$$

Where:
- $ x[n] $ is the input signal,
- $ y[n] $ is the output signal, and
- $ M $ is the number of points in the average.

For example, in a 5-point moving average filter, point 80 in the output signal is given by:

$$
y[80] = \frac{1}{5} (x[78] + x[79] + x[80] + x[81] + x[82])
$$

### Figure 15-1
- (a) A pulse buried in random noise.
- (b) and (c) Show the smoothing action of the moving average filter, which decreases the amplitude of the random noise (good), but also reduces the sharpness of the edges (bad).

The amount of noise reduction is equal to the square-root of the number of points in the average. For example, a 100-point moving average filter reduces the noise by a factor of 10.

[Watch the video explanation here](https://www.youtube.com/watch?v=tMvrnlf0eIA)

To understand why the moving average is the best solution, imagine we want to design a filter with a fixed edge sharpness. For example, let’s assume we fix the edge sharpness by specifying that there are eleven points in the rise of the step response. This requires that the filter kernel have eleven points. The optimization question is: how do we choose the eleven values in the filter kernel to minimize the noise on the output signal?

Since the noise we are trying to reduce is random, none of the input points is special; each is just as noisy as its neighbour. Therefore, it is useless to give preferential treatment to any one of the input points by assigning it a larger coefficient in the filter kernel. The lowest noise is obtained when all the input samples are treated equally, i.e., the moving average filter.

**Note:** Later in this chapter we show that other filters are essentially as good. The point is, no filter is better than the simple moving average.

https://kgptalkie.com/fir-filter-moving-average-filter-in-matlab