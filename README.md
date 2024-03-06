# Human-Activity-Recognition
The Human Activity Recognition(HAR) model differentiates the different activities a human performs. It has very
wide applications, such as in the field of healthcare to monitor the
physical activity of patients, in the field of sports and fitness to
analyze the activities of athletes to improve their performance,
in security, and many more. Here we tackle the
challenge of classifying human activities utilizing a range of
machine learning techniques, like Kernelized SVMs and different
architectures of Convolutional Neural Networks (CNNs). <br><br>

## About the Dataset
In this study, we work with two essential components of
our project: Image data and Associated labels, aiming to
classify them from a list of 15 actions. These include sitting,
clapping, dancing, using laptop, hugging, sleeping, drinking,
calling, cycling, laughing, fighting, eating, listening to music,
texting and running. <br>
There are a total of 12600 images in the training folder
of the dataset, along with an additional 5400 testing images.
Each of the 15 classes in the training set accounts for 6.7%
of the total images. This means there are 840 images per class. <br>
 
Link to the dataset: https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset  

![Screenshot 2024-03-05 181240](https://github.com/arjit06/Human-Activity-Recognition/assets/108218688/ea964500-ce60-4cc9-aceb-7493caa2bf73) 



## <br>EDA and Preprocessing <br>
### Class Imbalance
![bar_plot_class_imbalance](https://github.com/arjit06/Human-Activity-Recognition/assets/108218688/10aa0a2c-2768-40ff-86ac-3c67db34a771)

All the 15 classes in the HAR dataset have the same number
of images, hence, there will be no class imbalance based
on the no. of images per class. This will ensure balanced
training (which can help the model learn the characteristics
of all classes effectively) and reduced bias. <br> <br>


### Data Distribution
![piechart](https://github.com/arjit06/Human-Activity-Recognition/assets/108218688/71acbaa3-ae74-437c-b520-21f78b750c31)

There are a total of 12600 images in the training folder
of the dataset, along with an additional 5400 testing images.
Each of the 15 classes in the training set accounts for 6.7%
of the total images. This means there are 840 images per class.<br> <br>

### Histogram plots for all data and per class based (RGB and Grayscale)
![whole_data_hist](https://github.com/arjit06/Human-Activity-Recognition/assets/108218688/257d837c-76bc-46e8-b5ec-58c390419beb)

![histograms2](https://github.com/arjit06/Human-Activity-Recognition/assets/108218688/fbd3b4c8-9c3b-47cc-b7af-711bb9d908d2)

The dataset clearly doesn’t follow a gaussian distribution
as evident with strange peaks at the beginning and end (These
suggest the presence of ouliers and noise which needs to be
removed ) . Also on an average the colour distribution is even
, evident from the closely grouped peaks and modes.<br> <br>

### Histogram of oriented Gradients (HOG) 
![hog](https://github.com/arjit06/Human-Activity-Recognition/assets/108218688/cfe22319-57db-4e66-b1f7-be9b0e5253bc)

Histogram of Oriented Gradients, also known as HOG,
is a feature descriptor. It is used in computer vision and
image processing for the purpose of object detection. The
technique counts occurrences of gradient orientation in the
localized portion of an image. For the regions of the image,
it generates histograms using the magnitude and orientations
of the gradient.<br> <br>

### Edge map using Sobel’s filter
![sobels](https://github.com/arjit06/Human-Activity-Recognition/assets/108218688/6dc5221b-73ac-41ab-950e-2eab737f383d)

Sobel’s Edge detection filter is used to detect edges in the
images. These can be converted to statistical features like
mean, median, etc. to convert into feature vectors.<br><br>

### Local Binary Patterns (LBP) 
![LBP](https://github.com/arjit06/Human-Activity-Recognition/assets/108218688/cbaedbd1-e9dd-4582-b49b-5fe5bf2b478d)

LBPs compute a local representation of texture. This local
representation is constructed by comparing each pixel with
its surrounding neighborhood of pixels. LBP looks at points
surrounding a central point and tests whether the surrounding
points are greater than or less than the central point (i.e.,
gives a binary result). <br><br>
 
### Gaussian and Bilateral Filters
<img width="665" alt="Bilateral" src="https://github.com/arjit06/Human-Activity-Recognition/assets/108218688/0d1fc229-4ff1-4dc3-852d-c61448c67ba8">

A Gaussian filter is used to remove random noise from the
image. If applied with a large sigma, it starts to remove edge
information beyond removing noise. (i.e. it blurs).
The Bilateral Filter, a good image processing technique,
excels in noise reduction while preserving critical image
features and edges. It combines spatial proximity and pixel
intensity similarity to achieve selective smoothing without
compromising edge sharpness. In our case, it worked better
than Gaussian for noise because Bilateral filters are often
preferred in scenarios where preserving edge details is crucial. <br><br><br>


##  RESULTS FOR CLASSICAL ML APPROACHES
Here are the results for classical ML based techniques. The
baseline model used is SVMs (experimented with different
configurations of kernels and other parameters).<br><br>

| Model Architecture  | Accuracy |
| ------------- | ------------- |
| HOG + HSV + LBP features  | 35%  |
| Bilateral Filter (for noise removal), Sobel filter <br> (for BG separation) then HOG + HSV + LBP features | 30%  |
| Ensemble of HOG, HSV  | 31%  |
| SIFT Features  | 25%  |
| HSV+LBP  | 30%  |  

<br>

### Some Intermediate Prediction Results 
![image](https://github.com/arjit06/Human-Activity-Recognition/assets/108218688/5bf2de41-c3f6-47c7-8ef4-802735982d7e) 

<br><br>



### Performance Evaluation of Various Kernels for Optimal Model Selection
<ul>
<li> Linear Kernel: 28% </li>
<li> RBF Kernel (Gaussian): 34% </li> 
<li> Polynomial Kernel (Deg-6): 35% </li>
<li> Sigmoid Kernel: 14% </li>
</ul>
<br><br><br>

## FEEDBACK AND ISSUES
<b>Noise in the dataset:</b> Traditional feature extraction methods involve handcrafted
operations on the image. While these methods (filters and
thresholds) may capture certain local features, they do not
perform noise removal in a learned and adaptive way as they
are manual. Ties in the dataset added to the woes. <br>
 
<b>Imperfect feature extraction:</b> Handcrafted techniques like HOG, HSV, and LBP have
limited capacity to represent and generalize complex and
high-dimensional patterns inherent in human activities. <br>

Thus, traditional ML techniques and non-deep learningbased feature engineering do not lead to high accuracy in
this task. CNNs are capable of learning hierarchical
representations of data and reducing noise inherently due to
multiple layers. <br><br><br>

## DL BASED TECHNIQUES

Here the baseline model used is CNN. Different
architectures have been used including some pretrained
models that were fine tuned on the dataset. <br>

### Custom CNN based architecture ( using modified version of INCEPTION BLOCKS )
![Screenshot 2024-03-05 180743](https://github.com/arjit06/Human-Activity-Recognition/assets/108218688/34381340-c0ce-4b84-a188-2cafddb6a8cb)

We used 6 such inception blocks followed by a global avg
pooling at the end and then a fully connected layer for final
classification. After each inception block we did a maxpool
having stride=2 (for downsampling the image) . The no. of
channels were doubled after each block starting from 32
and all the way upto 512. The Activation function used was
ReLU followed by a Cross Entropy Loss. <br>
The following were the parameters used:-
<ul>
<li>Batch Size: 32 </li>
<li>No. of Epochs: 40 </li>
<li>Learning Rate: 0.001 </li>
<li>Image Size: 160x160 </li>
<li>Optimizer: Adam</li>
</ul>
<b>Accuracy on Cross-validation : 56%</b>
<br><br>

###  Pre-Trained VGG -16
![Screenshot 2024-03-05 190325](https://github.com/arjit06/Human-Activity-Recognition/assets/108218688/f0214cc0-7e21-4c98-9cb3-a08a8fbba064)

VGG-16 is a widely recognized convolutional neural
network architecture featuring 16 layers, including 13
convolutional layers and 3 fully connected layers. It utilizes
small 3x3 convolutional filters throughout the network,
enabling it to learn complex image features effectively.
Despite its simplicity, VGG-16 achieves strong performance
in image classification tasks and serves as a foundational
model in deep learning research and applications.

<b>Accuracy on Cross-validation : 53%</b>
<br><br>

###  Pre-Trained Efficient Net B7
![Screenshot 2024-03-05 190814](https://github.com/arjit06/Human-Activity-Recognition/assets/108218688/0982ace1-5f11-403c-b75b-8c33b1ee6844)

EfficientNet B7 is a part of the EfficientNet family,
renowned for its compound scaling strategy, which
systematically adjusts model depth, width, and resolution to
achieve optimal performance. As one of the largest variants,
B7 boasts deeper, wider, and higher-resolution architecture
compared to its counterparts. This design allows it to capture
intricate features from images efficiently. By leveraging
pre-trained weights from datasets like ImageNet, EfficientNet
B7 accelerates training and ensures robust performance across
various computer vision applications, making it an invaluable
tool in the field.

<b>Accuracy on Cross-validation ~ 70%</b>
<br><br><br>


## Conclusion
In this project the research of different machine learning
methods used to recognize human activities has been performed. In the first part a lot of different EDA techniques
from classical machine learning have been used for feature
extraction with the basline model as SVM. Then in the second
part different architectures of CNNs were employed to achieve
the result. This paper provides the comparison study of the
mentioned methods for human activity recognition.
The average accuracy of image classification using SVMs is
lower and approaches 30%. The application of different CNN
architectures has revealed higher accuracy results, although
EfficientNet B7 has reached around 70% average accuracy,
which indicates the best score of all applied methods. Considering the obtained results, further studies are needed to analyze
the eligibility of different and newly created CNN architectures
for the solution of image-based human activity classification
problem. <br><br><br>

## SOME FINAL PREDICTION RESULTS

![Screenshot 2024-03-05 191814](https://github.com/arjit06/Human-Activity-Recognition/assets/108218688/7e7007bd-46da-4709-aa71-89ce3014ef93)
















