# Tomato Leaf Disease Detection with PKD Framework
## Description
This GitHub repository contains the implementation of the Progressive Knowledge Distillation (PKD) framework for tomato leaf disease detection as described in the research paper "PKD: Progressive Knowledge Distillation Facilitating Tomato Leaf Disease Detection" by Hongmin Zhaoa and Jinzhou Xiea.
The PKD framework aims to address the challenge of deploying complex deep learning models for tomato leaf disease detection on resource-constrained terminal devices. It achieves this by gradually transferring knowledge from ahoven!> There was an error generating the text for this section. Please try again. If you want to know more about README.md file formatting and content, you can refer to the following resources:
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
  - [CSPDarkNet53 in YOLOv7](#cspdarknet53-in-yolov7)
  - [ResNet50 in Faster RCNN](#resnet50-in-faster-rcnn)
- [PKD Framework Modules](#pkd-framework-modules)
  - [Attention Distillation](#attention-distillation)
  - [Non-Local Module and Adaptive Feature Matching Module](#non--local-module-and-adaptive-feature-matching-module)
  - [Head Feature Distiller](#head-feature-distiller)
  - [Sliding Slice Cross Entropy](#sliding-slice-cross-entropy)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage Examples](#usage-examples)
- [License](#license)
## Installation
1.  Clone the repository:
```python
git clone https://github.com/your_username/PKD.git
```
2. Navigate to the cloned directory:
```python
cd PKD
```
3.  Install the required dependencies. The code is based on PyTorch, so make sure you have PyTorch installed. You can install other required packages using  `pip`:
```python
pip install -r requirements.txt
```
## Dataset
The experiments in the paper were conducted on the Kaggle tomato leaf disease dataset, which contains 21,328 images across 10 categories. You can download the dataset from the Kaggle website ([https://www.kaggle.com/code/samanfatima7/tomato-leaf-disease-94-accuracy](https://www.kaggle.com/code/samanfatima7/tomato-leaf-disease-94-accuracy)). After downloading, the dataset should be organized in a specific structure. For example, you might have a directory structure like this:
```python
data/ 
	train/ 
		category_1/ 
			image_1.jpg 
			image_2.jpg 
			... 
		category_2/ 
			... 
	val/ 
		category_1/ 
			... 
		category_2/ 
			... 
	test/ 
		category_1/ 
			... 
		category_2/ 
			...
```
## Model Architectures

  

### CSPDarkNet53 in YOLOv7

  

-   The  `CSPDarkNet53`  class in the code defines the architecture of the CSPDarkNet53 network. It consists of multiple convolutional layers and residual blocks. The network is designed to extract features from the input images.
-   In the  `YOLOv7`  class,  `CSPDarkNet53`  is used as the backbone network. The forward pass of the  `YOLOv7`  model first passes the input through the  `CSPDarkNet53`  backbone to obtain features, and then further processes these features in the neck and head sections (which need to be fully implemented according to the YOLOv7 architecture).

  

### ResNet50 in Faster RCNN

  

-   The  `FasterRCNNWithResNet50`  class uses the pre-trained  `ResNet50`  model from PyTorch as the backbone network.  `ResNet50`  is a well-known deep learning architecture that extracts hierarchical features from the input images.
-   After passing the input through the  `ResNet50`  backbone, the features are further processed in the RPN (Region Proposal Network), RoI (Region of Interest) pooling, classification, and regression head sections of the Faster RCNN model (which also need to be fully implemented according to the Faster RCNN architecture).

  

### PKD Framework Modules

  

#### Attention Distillation

  

-   The attention distillation module in the code combines spatial attention distillation and channel attention distillation techniques.
-   The  `ChannelAttention`  class implements the channel attention mechanism. It uses global average pooling and global max pooling to generate channel attention maps, which are then processed through fully connected layers and activation functions to obtain channel attention weights.
-   The  `SpatialAttention`  class implements the spatial attention mechanism. It uses convolution on the concatenated results of global average pooling and global max pooling along the channel dimension to obtain spatial attention weights.
-   The knowledge transfer process for both spatial and channel attention is implemented in a way that calculates the L2 loss between the teacher and student attention maps and uses this loss to update the student model's parameters.

  

#### Non - Local Module and Adaptive Feature Matching Module

  

-   The  `NonLocalModule`  class implements the Non - Local module. It uses convolutional layers to compute theta, phi, and psi values and then calculates an attention weight matrix based on these values. The output of the Non - Local module is a weighted sum of the input features, adjusted by a learnable scaling coefficient gamma.
-   The  `AFM`  class (Adaptive Feature Matching Module) adapts the features of the student network to match those of the teacher network. It uses linear transformations (theta, phi, and psi) to compute Q, K, and V values and then calculates an attention weight matrix. The output of the AFM is the sum of the original input features and a weighted version of the features, adjusted by the gamma coefficient.

  

#### Head Feature Distiller

  

-   The  `HFD`  class (Head Feature Distiller) aims to help the student model better understand and distinguish different tomato disease categories. It calculates a total loss that combines cross-entropy losses for class scores and object confidence, along with a WIoU loss for bounding box information.
-   The cross-entropy losses are calculated using the  `F.cross_entropy`  function in PyTorch, and the WIoU loss is calculated according to the formula described in the paper. The weights for each loss term (omega1, omega2, and omega3) are learnable parameters.

  

#### Sliding Slice Cross Entropy

  

-   The  `sliding_slice_cross_entropy`  function implements the Sliding Slice Cross Entropy (SSCE) mechanism. It first sorts the teacher and student model outputs and then slices them according to a specified window size and step size. The cross-entropy loss is calculated for each slice and then accumulated to obtain the final SSCE loss.

  

## Training

  

1.  Prepare the dataset as described in the  Dataset  section.
2.  Set the appropriate hyperparameters for training. These include parameters for the PKD framework such as the window size (w) and step size (s) for SSCE, weights for the HFD losses (omega1, omega2, omega3), and other training-related parameters such as learning rate, batch size, and number of epochs.
3.  Use an appropriate optimizer. In the code, you can choose from popular optimizers like Adam or SGD. For example, if you choose Adam:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```
4.  Define the loss function. The total loss for the PKD framework is a combination of losses from different modules as described in the paper. You can use the implemented loss functions for each module and sum them up to get the total loss. For example:
```python
total_loss = channel_attention_loss + spatial_attention_loss + non_local_loss + hfd_loss + ssce_loss
```
5.  Start the training loop. In each iteration of the loop, pass the input data through the teacher and student models, calculate the losses, and update the model parameters using the optimizer. A simplified example of a training loop is as follows:
```python
for epoch in range(num_epochs): 
	for batch in train_loader: 
		inputs, targets = batch 
		teacher_outputs = teacher_model(inputs) 												   		
		student_outputs = student_model(inputs)
		loss = calculate_total_loss(teacher_outputs,student_outputs) 
		optimizer.zero_grad() 
		loss.backward() 
		optimizer.step()
```
## Evaluation

  

1.  After training, you can evaluate the performance of the model on a validation or test set.
2.  Calculate relevant metrics such as mean Average Precision (mAP), Average Precision at different IoU thresholds (AP50, AP75, etc.), and classification accuracy. You can use functions and libraries in PyTorch to calculate these metrics. For example, to calculate mAP:
```python
from torchmetrics.detection.mean_ap import MeanAveragePrecision
   metric = MeanAveragePrecision()
   preds = model(test_data)
   metric.update(preds, test_targets)
   mAP = metric.compute()
```
3.  Compare the performance of the trained model with other models or baselines. The code repository can be used to reproduce the experiments in the paper and compare the results with the reported values.
## Usage Examples

  

1.  **Inference on a single image**:
    -   You can use the trained model to perform inference on a single tomato leaf disease image. First, load the model and the image:
```python
model = load_trained_model()
     image = load_image('path/to/image.jpg')
```
where  `load_trained_model`  is a function to load the trained model weights and  `load_image`  is a function to load the image and preprocess it appropriately.

  

-   Then, pass the image through the model and get the predictions:
```python
predictions = model(image)
```
The predictions will typically include information about the detected diseases, such as class probabilities and bounding box coordinates.
2.  **Inference on a batch of images**:

  

-   Similar to single image inference, you can load a batch of images and pass them through the model. For example:
```python
batch_images = load_batch_images('path/to/batch/images')
     predictions = model(batch_images)
```
where `load_batch_images` is a function to load a batch of images and preprocess them.
  

This project is licensed under the [License Name] license. Please refer to the LICENSE file in the repository for more details.

  

You can also add the following to the  `Description`  field of your GitHub repository:

  

This repository contains the code implementation of the PKD framework for tomato leaf disease detection. It includes the necessary code for the model architectures (CSPDarkNet53 in YOLOv7 and ResNet50 in Faster RCNN), as well as the implementation of all PKD framework modules such as attention distillation, Non - Local module, Adaptive Feature Matching Module, Head Feature Distiller, and Sliding Slice Cross Entropy. The code is accompanied by detailed documentation in the README file, which explains how to install the project, prepare the dataset, train the models, evaluate their performance, and use them for inference. Contributions to the project are encouraged to further improve and expand the functionality of the framework.

  
