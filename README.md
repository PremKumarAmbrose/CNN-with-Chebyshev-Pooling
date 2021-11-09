# CNN-with-Chebyshev-Pooling

A CNN  based image classifier in which an alternative layer for the Pooling is used, which is called Chebyshev Pooling layer.

when performing maxpooling in convolutional Neural Networks the global information is lost and only the local features are retained. This makes the decision process of tasks like image classification difficult. This chebyshev pooling layer makes use of Chebyshev's inequality to produce results about the probability distributions within the kernel which contains the functions of maximum nad average pooling. This layer ensures an output in the range of (0.1, 1.0) which is more stable for subsequent processing.


The model which is quantized to "integer only" using tensorflowlite's post quantization can be deployed into STM32 microcontrollers family with the help of XCubeAI of STMicroelectronics. 

[Link to XCubeAI Guide](https://www.st.com/resource/en/user_manual/dm00570145-getting-started-with-xcubeai-expansion-package-for-artificial-intelligence-ai-stmicroelectronics.pdf )

**References**
K. -H. Chan, G. Pau and S. -K. Im, "Chebyshev Pooling: An Alternative Layer for the Pooling of CNNs-Based Classifier," 2021 IEEE 4th International Conference on Computer and Communication Engineering Technology (CCET), 2021, pp. 106-110, doi: 10.1109/CCET52649.2021.9544405.

[Link to Research Paper](https://ieeexplore.ieee.org/document/9544405)
