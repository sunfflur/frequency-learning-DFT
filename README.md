# A frequency-domain approach with learnable filters for image classification

## Abstract
#### Machine learning applied to computer vision and signal processing is achieving results comparable to the human brain due to the great improvements brought by deep neural networks (DNN). The majority of state-of-the-art architectures are DNN related, but only a few explicitly explore the frequency domain to extract useful information and improve the results. This paper presents a new approach for exploring the Fourier transform of the input images, which is composed of trainable frequency filters that boost discriminative components in the spectrum. Additionally, we propose a cropping procedure to allow the network to learn both global and local spectral features of the image blocks. The proposed method proved to be competitive with respect to well-known DNN architectures in the selected experiments, which involved texture classification, cataract detection and retina image analysis, where there is a noticeable appeal for the frequency domain, with the advantage of being a lightweight model.

## Experiments
#### Repository composed by the algorithms used to compare the frequency-domain approach and some ConvNets on 4 datasets:
  - Kylberg Texture Dataset;
  - Cataract Detection;
  - Ocular Disease Recognition;
  - EyeQ.

## Usage

In each folder, you'll find individual Python notebooks that can be executed independently. The notebooks cover various aspects of the project and can be used for different purposes. Follow the content description within each notebook to understand and utilize the code effectively. It's important to mention the following notebooks refers to the implementation of the frequency-based model on the Fourier domain.

## Content description

| [Kylberg Texture Dataset](https://kylberg.org/kylberg-texture-dataset-v-1-0/) | Description |
|-------------|-------------|
| Exp_02_ViTb6_32.pynb    | This notebook contains the implementation of ViT-b/16 and ViT-b/32 to classify the 28 classes from Kylberg Texture Dataset. There are two variations of ViT-b/16: using image size 224x224 and 384x384 pixels. |
| Exp_02_a_**ConvNet**.pynb    | These types of notebooks presents the implementation of the **ConvNet** on Kylberg dataset considering the use of transfer learning -- training only the output layer -- and random initialization -- training all the network from scratch --. |
| Exp_02_a_**ConvNet**\_varyingdatasize.pynb    | These types of notebooks presents the implementation of the **ConvNet** on Kylberg dataset similarly to the files above, however, in the _Random Initialization_ section, it also computes the test accuracy when the network is dealing with different sizes of training data -- from 100% to 10% of all the training set. |
| Exp_02_a.pynb   | This notebook contains the implementation of the frequency-based model on Kylberg dataset. It considers only 3 levels of blocks division and width = 1. |
| Exp_02_a_v2.pynb   | This notebook is an updated version of the above. It contains ablation studies in some components of the proposed approach: the number of splitting levels vary from 3 to 1 and the width vary from 1 to 4. |

| [EyeQ Dataset](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data) | Description |
|-------------|-------------|
| Exp_EyeQ.pynb   | This notebook contains the implementation of the frequency-based model on both classes of EyeQ dataset: _good_ and _reject_. It considers 3 levels of blocks division and width = 1. |
| Exp_EyeQ_**ConvNet**.pynb    | These types of notebooks presents the implementation of the **ConvNet** on EyeQ dataset considering the use of transfer learning -- training only the output layer -- and random initialization -- training all the network from scratch --. |

| [ODIR Dataset](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k) | Description |
|-------------|-------------|
| Exp_ODR2_a.pynb   | This notebook contains the implementation of the frequency-based model on the classes _cataract_ and _normal_ from the Ocular Disease Intelligent Recognition (ODIR) dataset. It considers 3 levels of blocks division and width = 1. To avoid possible bias, we also take care of keeping data from the same patient in the same set, since some of the patients have the disease on both eyes. |
| Exp_ODR2_**ConvNet**.pynb    | These types of notebooks presents the implementation of the **ConvNets** on ODIR dataset considering the use of transfer learning -- training only the output layer -- and random initialization -- training all the network from scratch --. |

| [Cataract Dataset](https://www.kaggle.com/datasets/jr2ngb/cataractdataset) | Description |
|-------------|-------------|
| Exp_Retina.pynb   | This notebook contains the implementation of the frequency-based model on the classes _cataract_ and _normal_ from the Kaggle Cataract dataset. It considers 3 levels of blocks division and width = 1. This experiment is similar to the ODIR related experiment, however, here we do not consider the fundus images of the same patients in the same set. |
| Exp_Retina_**ConvNet**.pynb    | These types of notebooks presents the implementation of the **ConvNets** on Kaggle Cataract dataset considering the use of transfer learning -- training only the output layer -- and random initialization -- training all the network from scratch --. |



