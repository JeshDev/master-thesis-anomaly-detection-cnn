# Anomaly Detection in Additive Manufacturing using Deep Learning

By [Jeshwanth Pilla](https://www.linkedin.com/in/jeshwanth-p/)

## Contents
0. [Introduction](#introduction)
0. [Quick Guide](#quick-guide)
0. [Results](#results)
0. [Conclusion](#conclusion)


## Introduction 

With the increasing adoption of Additive Manufacturing in the industries, quality assurance has become a critical and increasingly challenging process. Especially in an industrial setup, in-situ monitoring is crucial to facilitate faster prototyping and industrialisation. Towards this objective, there have been increasing efforts towards real-time detection of recoating anomalies that occur in Laser Powder Bed Fusion. These defects are observed to have a detrimental effect on the quality of the final product. Machine learning methods have seen great success in addressing this problem; however, it usually involves crafting features specific to each printer type. This thesis explores how deep learning, particularly Convolutional Neural Networks, could be leveraged for further performance. A deep regression model is formulated to predict a score defining the quality of recoating at each layer from the imaging data made available through inbuilt monitoring solutions. The presented methodology is built by employing transfer learning with recent state-of-the-art architectures and demonstrated robust performance with reduced computation effort. These models also showed generalizability to changes in illumination conditions observed with different machines.

## Quick Guide

The models are developed using Python 3.7, Tensorflow 2.3.1, Keras 2.4.0 and relevant libraries such as Numpy, Pandas, PIL. The training is performed on Nvidia Tesla P40 GPU
with 24 GB RAM.

### TensorFlow
The model is constructed in ```cnn_networks.py``` and pretrained ImageNet weights are used for training.<br />
The code for training  and generating predictions is found in ```cnn_regressor.py``` and and hyperparameters can be changed there. Use ```python cnn_regressor.py csv_file image_patches_path model_network no_of_epochs batch_size learning_rate``` to try the code.<br />

## Results

In the following table, the performance comparision of different CNN networks is reported below

| Model     |  Test MSE  |  Rsquare  | 
|-----------------------------|:-----:|:-----:|
| ResNet50		| **0.0067** | **0.916** | 
| MobileNetV2	| 0.0085| 0.903 | 

## Conclusion

This thesis proposed a deep regression model that can successfully detect the surface-visible anomalies observed during the Powder Bed monitoring. It has improved upon the previous machine learning methodologies by employing more contemporary architectures with less complexity and further used transfer learning to increase computational efficiency. The deep learning approach also shows more potential in learning representations than machine learning methods obviating any effort for feature engineering and contextual heuristics. The novelty of this approach is using a regression model for powder bed anomaly detection as previous work usually followed a Classification approach. 
This procedure has also reduced the manual labelling effort involved and prevented the issue of class imbalances typically observed. Furthermore, the usage of Convolutional Neural Networks for regression is also slightly distinctive.

The results reinforce the hypothesis that predominantly a general-purpose network (ResNet-50 or MobileNetV2) adequately tuned can yield results close to the state-of-the-art without resorting to more complex and ad-hoc models. These models successfully extracted features from grayscale images, although these networks are usually optimised to learn from images with all colour channels. The efficacy of the previous feature-based machine learning model was also implicitly assessed. An impressive development is discovered that deep learning models can make the correct predictions even with small scale inconsistencies in the ground truth. This is validated by a training procedure that could achieve decent performance with deep learning models even in the presence of label noise.

The presented methodology can also be tuned further to enable its implementation in a real-time environment. The robustness of anomaly detection models to illumination changes,
which is relatively underexplored, is also investigated in this study. This result shows promise that models trained with image data of one AM printer type could be used for another and reduce the need for machine-specific calibration or lighting configuration adjustments. It could also save a lot of training and data collection effort.
