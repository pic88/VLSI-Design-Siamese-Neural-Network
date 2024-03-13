# VLSI Design: Siamese Neural Network
 SSN Architectural Design (verilog)
 use DW Lib
## 架構圖
![image](https://hackmd.io/_uploads/rJhm3aETT.png)
## 架構
- input image 2x(4x4x3)images
- input Kernel  (3x3x3)
- input weight  (2x2)
- padding image 4x4x3 -> 6x6x3 (Replication Padding || Zero Padding)
- Convolution (image+Kernel) => FeatureMap(4x4)
- Max-Pooling (4x4)->(2x2)
- Fully Connected & Flatten (input x weight)
- Min-Max Normalization
- Two Activation Function
- ![image](https://hackmd.io/_uploads/B1Fky0EaT.png)
- output L1 distance
