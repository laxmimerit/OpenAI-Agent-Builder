https://kgptalkie.com/system-identification-using-adaptive-lms-normalized-lms-filter-in-matlab

# System Identification using Adaptive LMS and Normalized LMS Filter in MATLAB

Published by  
on  
11 September 2016  
11 September 2016  

There are four major types of adaptive filtering configurations:  
- Adaptive system identification  
- Adaptive noise cancellation  
- Adaptive linear prediction  
- Adaptive inverse system  

All of the above systems are similar in the implementation of the algorithm but different in system configuration. All 4 systems have the same general parts:  
- Input $ x(n) $  
- Desired result $ d(n) $  
- Output $ y(n) $  
- Adaptive transfer function $ w(n) $  
- Error signal $ e(n) $, which is the difference between the desired output $ u(n) $ and the actual output $ y(n) $  

In addition to these parts, the system identification and the inverse system configurations have an unknown linear system $ u(n) $ that can receive an input and give a linear output to the given input.

## Adaptive System Identification Configuration

The adaptive system identification is primarily responsible for determining a discrete estimation of the transfer function for an unknown digital or analog system. The same input $ x(n) $ is applied to both the adaptive filter and the unknown system from which the outputs are compared (see figure 1). The output of the adaptive filter $ y(n) $ is subtracted from the output of the unknown system resulting in a desired signal $ d(n) $.

The resulting difference is an error signal $ e(n) $ used to manipulate the filter coefficients of the adaptive system trending towards an error signal of zero. After a number of iterations of this process are performed, and if the system is designed correctly, the adaptive filter’s transfer function will converge to, or near to, the unknown system’s transfer function. For this configuration, the error signal does not have to go to zero, although convergence to zero is the ideal situation, to closely approximate the given system. There will, however, be a difference between adaptive filter transfer function and the unknown system transfer function if the error is nonzero and the magnitude of that difference will be directly related to the magnitude of the error signal.

Additionally, the order of the adaptive system will affect the smallest error that the system can obtain. If there are insufficient coefficients in the adaptive system to model the unknown system, it is said to be under specified. This condition may cause the error to converge to a nonzero constant instead of zero. In contrast, if the adaptive filter is over specified, meaning that there are more coefficients than needed to model the unknown system, the error will converge to zero, but it will increase the time it takes for the filter to converge.

For more details, refer to the original article:  
https://kgptalkie.com/system-identification-using-adaptive-lms-normalized-lms-filter-in-matlab