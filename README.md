## A frequency-domain approach with learnable filters for image classification

### Abstract
#### Machine learning applied to computer vision and signal processing is achieving results comparable to the human brain due to the great improvements brought by deep neural networks (DNN). The majority of state-of-the-art architectures are DNN related, but only a few explicitly explore the frequency domain to extract useful information and improve the results. This paper presents a new approach for exploring the Fourier transform of the input images, which is composed of trainable frequency filters that boost discriminative components in the spectrum. Additionally, we propose a cropping procedure to allow the network to learn both global and local spectral features of the image blocks. The proposed method proved to be competitive with respect to well-known DNN architectures in the selected experiments, which involved texture classification, cataract detection and retina image analysis, where there is a noticeable appeal for the frequency domain, with the advantage of being a lightweight model.

### Experiments
#### Repository composed by the algorithms used to compare the frequency-domain approach and some CNNs on 4 datasets:
  - Kylberg Texture Dataset;
  - Cataract Detection;
  - Ocular Disease Recognition;
  - EyeQ.
