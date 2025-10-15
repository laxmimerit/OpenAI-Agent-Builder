https://kgptalkie.com/find-the-periodicity-of-noisy-data-using-autocorrelation-method-in-matlab

# Find the Periodicity of Noisy Data using Autocorrelation Method in MATLAB

Published by  
on  
10 September 2016  
10 September 2016  

## What is Autocorrelation?

Autocorrelation (short ACF, autocorrelation function) is a cross-correlation of a signal with itself. By correlating a signal with itself, repetitive patterns will stand out and make it much easier to see.

The (discrete) autocorrelation of a signal $ x $ is defined by the following simple equation:

$$
R[j] = \sum_{n=0}^{N-1} x[n] \cdot x[n + j]
$$

Where:
- $ R[j] $ is the autocorrelation at lag $ j $
- $ x[n] $ is the original signal
- $ N $ is the length of the signal

### Key Concepts
- **$ R[0] $** is basically the **energy of the signal** and therefore the maximum of the ACF.
- **$ R[1] $** is the correlation of the signal with itself shifted by one sample.

### Detecting Periodicity
If the signal has a significant enough self-similarity, the ACF will show this relation. In the case of a suspected periodicity, the signal will repeat itself after each period. 

To detect periodicity:
1. Compute the autocorrelation of the signal
2. Check for maxima in the ACF result
3. If the maxima are within a certain delta of your suspected periodicity, the signal is periodic

> **Note**: This method is particularly useful for analyzing noisy data, as the autocorrelation helps to highlight underlying periodic patterns that may be obscured by noise.

For more details, refer to the original article:  
https://kgptalkie.com/find-the-periodicity-of-noisy-data-using-autocorrelation-method-in-matlab