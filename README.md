# Data augmentation
List of useful data augmentation resources. You will find here some links to more or less popular github repos :sparkles:, libraries, papers :books: and other information.

Do you like it? Feel free to :star: !
Feel free to pull request!

* [Introduction](https://github.com/AgaMiko/data-augmentation/blob/master/README.md#Introduction)
* [Common techniques](https://github.com/AgaMiko/data-augmentation/blob/master/README.md#Common-techniques)
* [Papers](https://github.com/AgaMiko/data-augmentation/blob/master/README.md#Papers)
* [Repositories](https://github.com/AgaMiko/data-augmentation/blob/master/README.md#Repositories)


![Enriching data illustration](https://image.slidesharecdn.com/data-thekeytodeeplearning-aitoday-braincreators-25thjuly2017-170726144816/95/data-the-key-to-deep-learning-39-638.jpg)
[Source](https://www.slideshare.net/braincreators/data-the-key-to-deep-learning)


# Introduction
## What is data augmentation?
Data augmentation can be simply described as any method that makes our dataset larger. To create more images for example, we could zoom the in and save a result, we could change the brightness of the image or rotate it. To get bigger sound dataset we could try raise or lower the pitch of the audio sample or slow down/speed up.

* Image
  * [Traditional transformations](https://github.com/AgaMiko/data-augmentation/blob/master/README.md#Traditional-transformations) - linear and elastic transformations. Most commonly used.
  * [Advanced transformations](https://github.com/AgaMiko/data-augmentation/blob/master/README.md#Advanced-transformations) - More advanced techniques used rarely, usually for specific purpose.
  * [Neural-based transformations](https://github.com/AgaMiko/data-augmentation/blob/master/README.md#Neural-based-transformations)
* Sound
* Text

# Common techniques
# Images
## Traditional transformations
Traditional transformations are the most common data augmentation methods applied in deep learning. Traditional transformations are mainly defined as affine (linear) and geometric (elastic) transformations. Typical example of linear operations on an image are rotation, shear, reflection, scaling, whereas geometric can include brightness manipulation, contrast change, saturation or hue.

## Advanced transformations
More advanced transformations include [adversarial training](http://openaccess.thecvf.com/content_cvpr_2018/html/Peng_Jointly_Optimize_Data_CVPR_2018_paper.html), [Random erasing](https://github.com/zhunzhong07/Random-Erasing),  [Weather transforms](https://github.com/albu/albumentations/blob/master/notebooks/example_weather_transforms.ipynb).

## Neural-based transformations
Relatively new branch of data augmentation methods. It uses neural networks to generate new data. The most popula are [Generative Adversarial Networks](https://ieeexplore.ieee.org/abstract/document/8363576) and [Neural Style Transfer](https://ieeexplore.ieee.org/document/8864616).

# Audio
Typical audio augmentations are mixing few sounds, add gaussian noise, speed up or slow down, shift pitch and shift. Less popular than image augmentation, probably because audio classification problems are often resolved with mel-spectrograms and convolutional neural networks.
 
# Text (NLP)
Not as popular and common as image augmentations due to advanced level of those augmentations: [Contextual data augmentation](https://github.com/pfnet-research/contextual_augmentation)

# Papers
In progress

## 2019
* [Style transfer-based image synthesis as an efficient regularization technique in deep learning](https://arxiv.org/abs/1905.10974); Agnieszka Mikołajczyk, Michał Grochowski; These days deep learning is the fastest-growing area in the field of Machine Learning. Convolutional Neural Networks are currently the main tool used for image analysis and classification purposes. Although great achievements and perspectives, deep neural networks and accompanying learning algorithms have some relevant challenges to tackle. In this paper, we have focused on the most frequently mentioned problem in the field of machine learning, that is relatively poor generalization abilities. Partial remedies for this are regularization techniques eg dropout, batch normalization, weight decay, transfer learning, early stopping and data augmentation. In this paper, we have focused on data augmentation. We propose to use a method based on a neural style transfer, which allows generating new unlabeled images of a high perceptual quality that combine the content of a base image with the appearance of another one. In a proposed approach, the newly created images are described with pseudo-labels, and then used as a training dataset. Real, labeled images are divided into the validation and test set. We validated the proposed method on a challenging skin lesion classification case study. Four representative neural architectures are examined. Obtained results show the strong potential of the proposed approach.
![mikolajczyk_st](images/mikolajczyk_st.PNG)

* [Generative adversarial network in medical imaging: A review](https://www.sciencedirect.com/science/article/abs/pii/S1361841518308430); Xin Yi, Ekta Walia, Paul Babyna; Generative adversarial networks have gained a lot of attention in the computer vision community due to their capability of data generation without explicitly modelling the probability density function. The adversarial loss brought by the discriminator provides a clever way of incorporating unlabeled samples into training and imposing higher order consistency. This has proven to be useful in many cases, such as domain adaptation, data augmentation, and image-to-image translation. These properties have attracted researchers in the medical imaging community, and we have seen rapid adoption in many traditional and novel applications, such as image reconstruction, segmentation, detection, classification, and cross-modality synthesis. Based on our observations, this trend will continue and we therefore conducted a review of recent advances in medical imaging using the adversarial training scheme with the hope of benefiting researchers interested in this technique.

* [Data Augmentation via Dependency Tree Morphing for Low-Resource Languages](https://arxiv.org/abs/1903.09460); Gözde Gül Şahin, Mark Steedman; Neural NLP systems achieve high scores in the presence of sizable training dataset. Lack of such datasets leads to poor system performances in the case low-resource languages. We present two simple text augmentation techniques using dependency trees, inspired from image processing. We crop sentences by removing dependency links, and we rotate sentences by moving the tree fragments around the root. We apply these techniques to augment the training sets of low-resource languages in Universal Dependencies project. We implement a character-level sequence tagging model and evaluate the augmented datasets on part-of-speech tagging task. We show that crop and rotate provides improvements over the models trained with non-augmented data for majority of the languages, especially for languages with rich case marking systems.

* [Data augmentation for instrument classification robust to audio effects](https://arxiv.org/abs/1907.08520); António Ramires, Xavier Serra; Reusing recorded sounds (sampling) is a key component in Electronic Music Production (EMP), which has been present since its early days and is at the core of genres like hip-hop or jungle. Commercial and non-commercial services allow users to obtain collections of sounds (sample packs) to reuse in their compositions. Automatic classification of one-shot instrumental sounds allows automatically categorising the sounds contained in these collections, allowing easier navigation and better characterisation. Automatic instrument classification has mostly targeted the classification of unprocessed isolated instrumental sounds or detecting predominant instruments in mixed music tracks. For this classification to be useful in audio databases for EMP, it has to be robust to the audio effects applied to unprocessed sounds. In this paper we evaluate how a state of the art model trained with a large dataset of one-shot instrumental sounds performs when classifying instruments processed with audio effects. In order to evaluate the robustness of the model, we use data augmentation with audio effects and evaluate how each effect influences the classification accuracy.


## 2018
* [Data augmentation for improving deep learning in image classification problem](https://ieeexplore.ieee.org/abstract/document/8388338); Agnieszka Mikołajczyk, Michał Grochowski; These days deep learning is the fastest-growing field in the field of Machine Learning (ML) and Deep Neural Networks (DNN). Among many of DNN structures, the Convolutional Neural Networks (CNN) are currently the main tool used for the image analysis and classification purposes. Although great achievements and perspectives, deep neural networks and accompanying learning algorithms have some relevant challenges to tackle. In this paper, we have focused on the most frequently mentioned problem in the field of machine learning, that is the lack of sufficient amount of the training data or uneven class balance within the datasets. One of the ways of dealing with this problem is so called data augmentation. In the paper we have compared and analyzed multiple methods of data augmentation in the task of image classification, starting from classical image transformations like rotating, cropping, zooming, histogram based methods and finishing at Style Transfer and Generative Adversarial Networks, along with the representative examples. Next, we presented our own method of data augmentation based on image style transfer. The method allows to generate the new images of high perceptual quality that combine the content of a base image with the appearance of another ones. The newly created images can be used to pre-train the given neural network in order to improve the training process efficiency. Proposed method is validated on the three medical case studies: skin melanomas diagnosis, histopathological images and breast magnetic resonance imaging (MRI) scans analysis, utilizing the image classification in order to provide a diagnose. In such kind of problems the data deficiency is one of the most relevant issues. Finally, we discuss the advantages and disadvantages of the methods being analyzed.

* [Augmentation Techniques for Mobile Cloud Computing: A Taxonomy, Survey, and Future Directions](https://dl.acm.org/citation.cfm?id=3152397); 	Bowen Zhou, Rajkumar Buyya; Despite the rapid growth of hardware capacity and popularity in mobile devices, limited resources in battery and processing capacity still lack the ability to meet increasing mobile users’ demands. Both conventional techniques and emerging approaches are brought together to fill this gap between user demand and mobile devices’ limited capabilities. Recent research has focused on enhancing the performance of mobile devices via augmentation techniques. Augmentation techniques for mobile cloud computing refer to the computing paradigms and solutions to outsource mobile device computation and storage to more powerful computing resources in order to enhance a mobile device’s computing capability and energy efficiency (e.g., code offloading). Adopting augmentation techniques in the heterogeneous and intermittent mobile cloud computing environment creates new challenges for computation management, energy efficiency, and system reliability. In this article, we aim to provide a comprehensive taxonomy and survey of the existing techniques and frameworks for mobile cloud augmentation regarding both computation and storage. Different from the existing taxonomies in this field, we focus on the techniques aspect, following the idea of realizing a complete mobile cloud computing system. The objective of this survey is to provide a guide on what available augmentation techniques can be adopted in mobile cloud computing systems as well as supporting mechanisms such as decision-making and fault tolerance policies for realizing reliable mobile cloud services. We also present a discussion on the open challenges and future research directions in this field.

* [Synthetic data augmentation using GAN for improved liver lesion classification](https://ieeexplore.ieee.org/abstract/document/8363576); Maayan Frid-Adar, Eyal Klang, Michal Amitai, Jacob Goldberger, Hayit Greenspan; In this paper, we present a data augmentation method that generates synthetic medical images using Generative Adversarial Networks (GANs). We propose a training scheme that first uses classical data augmentation to enlarge the training set and then further enlarges the data size and its diversity by applying GAN techniques for synthetic data augmentation. Our method is demonstrated on a limited dataset of computed tomography (CT) images of 182 liver lesions (53 cysts, 64 metastases and 65 hemangiomas). The classification performance using only classic data augmentation yielded 78.6% sensitivity and 88.4% specificity. By adding the synthetic data augmentation the results significantly increased to 85.7% sensitivity and 92.4% specificity.

* [Augmentation strategies for clozapine refractory schizophrenia: A systematic review and meta-analysis](https://journals.sagepub.com/doi/abs/10.1177/0004867418772351); Dan J Siskind, Michael Lee, Arul Ravindran, Qichen Zhang, Evelyn Ma, Balaji Motamarri, Steve Kisely; Although clozapine is the most effective medication for treatment refractory schizophrenia, only 40% of people will meet response criteria. We therefore undertook a systematic review and meta-analysis of global literature on clozapine augmentation strategies.We systematically reviewed PubMed, PsycInfo, Embase, Cochrane Database, Chinese Biomedical Literature Service System and China Knowledge Resource Integrated Database for randomised control trials of augmentation strategies for clozapine resistant schizophrenia. We undertook pairwise meta-analyses of within-class interventions and, where possible, frequentist mixed treatment comparisons to differentiate treatment effectiveness.
We identified 46 studies of 25 interventions. On pairwise meta-analyses, the most effective augmentation agents for total psychosis symptoms were aripiprazole (standardised mean difference: 0.48; 95% confidence interval: −0.89 to −0.07) fluoxetine (standardised mean difference: 0.73; 95% confidence interval: −0.97 to −0.50) and, sodium valproate (standardised mean difference: 2.36 95% confidence interval: −3.96 to −0.75). Memantine was effective for negative symptoms (standardised mean difference: −0.56 95% confidence interval: −0.93 to −0.20). However, many of these results included poor-quality studies. Single studies of certain antipsychotics (penfluridol), antidepressants (paroxetine, duloxetine), lithium and Ginkgo biloba showed potential, while electroconvulsive therapy was highly promising. Mixed treatment comparisons were only possible for antipsychotics, and these gave similar results to the pairwise meta-analyses.
On the basis of the limited data available, the best evidence is for the use of aripiprazole, fluoxetine and sodium valproate as augmentation agents for total psychosis symptoms and memantine for negative symptoms. However, these conclusions are tempered by generally short follow-up periods and poor study quality.

* [Alcoholism Detection by Data Augmentation and Convolutional Neural Network with Stochastic Pooling](https://link.springer.com/article/10.1007/s10916-017-0845-x); Shui-Hua Wang, Yi-Ding Lv, Yuxiu Sui, Shuai Liu, Su-Jing Wang, Yu-Dong Zhang; Alcohol use disorder (AUD) is an important brain disease. It alters the brain structure. Recently, scholars tend to use computer vision based techniques to detect AUD. We collected 235 subjects, 114 alcoholic and 121 non-alcoholic. Among the 235 image, 100 images were used as training set, and data augmentation method was used. The rest 135 images were used as test set. Further, we chose the latest powerful technique—convolutional neural network (CNN) based on convolutional layer, rectified linear unit layer, pooling layer, fully connected layer, and softmax layer. We also compared three different pooling techniques: max pooling, average pooling, and stochastic pooling. The results showed that our method achieved a sensitivity of 96.88%, a specificity of 97.18%, and an accuracy of 97.04%. Our method was better than three state-of-the-art approaches. Besides, stochastic pooling performed better than other max pooling and average pooling. We validated CNN with five convolution layers and two fully connected layers performed the best. The GPU yielded a 149× acceleration in training and a 166× acceleration in test, compared to CPU.

* [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501); Ekin D. Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, Quoc V. Le; Data augmentation is an effective technique for improving the accuracy of modern image classifiers. However, current data augmentation implementations are manually designed. In this paper, we describe a simple procedure called AutoAugment to automatically search for improved data augmentation policies. In our implementation, we have designed a search space where a policy consists of many sub-policies, one of which is randomly chosen for each image in each mini-batch. A sub-policy consists of two operations, each operation being an image processing function such as translation, rotation, or shearing, and the probabilities and magnitudes with which the functions are applied. We use a search algorithm to find the best policy such that the neural network yields the highest validation accuracy on a target dataset. Our method achieves state-of-the-art accuracy on CIFAR-10, CIFAR-100, SVHN, and ImageNet (without additional data). On ImageNet, we attain a Top-1 accuracy of 83.5% which is 0.4% better than the previous record of 83.1%. On CIFAR-10, we achieve an error rate of 1.5%, which is 0.6% better than the previous state-of-the-art. Augmentation policies we find are transferable between datasets. The policy learned on ImageNet transfers well to achieve significant improvements on other datasets, such as Oxford Flowers, Caltech-101, Oxford-IIT Pets, FGVC Aircraft, and Stanford Cars.

* [Data Augmentation by Pairing Samples for Images Classification](https://arxiv.org/abs/1801.02929); Hiroshi Inoue; Data augmentation is a widely used technique in many machine learning tasks, such as image classification, to virtually enlarge the training dataset size and avoid overfitting. Traditional data augmentation techniques for image classification tasks create new samples from the original training data by, for example, flipping, distorting, adding a small amount of noise to, or cropping a patch from an original image. In this paper, we introduce a simple but surprisingly effective data augmentation technique for image classification tasks. With our technique, named SamplePairing, we synthesize a new sample from one image by overlaying another image randomly chosen from the training data (i.e., taking an average of two images for each pixel). By using two images randomly selected from the training set, we can generate N2 new samples from N training samples. This simple data augmentation technique significantly improved classification accuracy for all the tested datasets; for example, the top-1 error rate was reduced from 33.5% to 29.0% for the ILSVRC 2012 dataset with GoogLeNet and from 8.22% to 6.93% in the CIFAR-10 dataset. We also show that our SamplePairing technique largely improved accuracy when the number of samples in the training set was very small. Therefore, our technique is more valuable for tasks with a limited amount of training data, such as medical imaging tasks.

* [Generalizing to Unseen Domains via Adversarial Data Augmentation](http://papers.nips.cc/paper/7779-generalizing-to-unseen-domains-via-adversarial-data-augmentation); Riccardo Volpi, Hongseok Namkoong, Ozan Sener, John C. Duchi, Vittorio Murino, Silvio Savarese; We are concerned with learning models that generalize well to different unseen domains. We consider a worst-case formulation over data distributions that are near the source domain in the feature space. Only using training data from a single source distribution, we propose an iterative procedure that augments the dataset with examples from a fictitious target domain that is "hard" under the current model. We show that our iterative scheme is an adaptive data augmentation method where we append adversarial examples at each iteration. For softmax losses, we show that our method is a data-dependent regularization scheme that behaves differently from classical regularizers that regularize towards zero (e.g., ridge or lasso). On digit recognition and semantic segmentation tasks, our method learns models improve performance across a range of a priori unknown target domains.

* [Feature Space Transfer for Data Augmentation](http://openaccess.thecvf.com/content_cvpr_2018/html/Liu_Feature_Space_Transfer_CVPR_2018_paper.html); Bo Liu, Xudong Wang, Mandar Dixit, Roland Kwitt, Nuno Vasconcelos; The problem of data augmentation in feature space is considered. A new architecture, denoted the FeATure TransfEr Network (FATTEN), is proposed for the modeling of feature trajectories induced by variations of object pose. This architecture exploits a parametrization of the pose manifold in terms of pose and appearance. This leads to a deep encoder/decoder network architecture, where the encoder factors into an appearance and a pose predictor. Unlike previous attempts at trajectory transfer, FATTEN can be efficiently trained end-to-end, with no need to train separate feature transfer functions. This is realized by supplying the decoder with information about a target pose and the use of a multi-task loss that penalizes category- and pose-mismatches. In result, FATTEN discourages discontinuous or non-smooth trajectories that fail to capture the structure of the pose manifold, and generalizes well on object recognition tasks involving large pose variation. Experimental results on the artificial ModelNet database show that it can successfully learn to map source features to target features of a desired pose, while preserving class identity. Most notably, by using feature space transfer for data augmentation (w.r.t. pose and depth) on SUN-RGBD objects, we demonstrate considerable performance improvements on one/few-shot object recognition in a transfer learning setup, compared to current state-of-the-art methods.

* [CamStyle: A Novel Data Augmentation Method for Person Re-Identification](https://ieeexplore.ieee.org/abstract/document/8485427); Zhun Zhong, Liang Zheng, Zhedong Zheng, Shaozi Li, Yi Yang; Person re-identification (re-ID) is a cross-camera retrieval task that suffers from image style variations caused by different cameras. The art implicitly addresses this problem by learning a camera-invariant descriptor subspace. In this paper, we explicitly consider this challenge by introducing camera style (CamStyle). CamStyle can serve as a data augmentation approach that reduces the risk of deep network overfitting and that smooths the CamStyle disparities. Specifically, with a style transfer model, labeled training images can be style transferred to each camera, and along with the original training samples, form the augmented training set. This method, while increasing data diversity against overfitting, also incurs a considerable level of noise. In the effort to alleviate the impact of noise, the label smooth regularization (LSR) is adopted. The vanilla version of our method (without LSR) performs reasonably well on few camera systems in which overfitting often occurs. With LSR, we demonstrate consistent improvement in all systems regardless of the extent of overfitting. We also report competitive accuracy compared with the state of the art on Market-1501 and DukeMTMC-re-ID. Importantly, CamStyle can be employed to the challenging problems of one view learning and unsupervised domain adaptation (UDA) in person re-identification (re-ID), both of which have critical research and application significance. The former only has labeled data in one camera view and the latter only has labeled data in the source domain. Experimental results show that CamStyle significantly improves the performance of the baseline in the two problems. Specially, for UDA, CamStyle achieves state-of-the-art accuracy based on a baseline deep re-ID model on Market-1501 and DukeMTMC-reID. Our code is available at: https://github.com/zhunzhong07/CamStyle.

* [Medical Image Synthesis for Data Augmentation and Anonymization Using Generative Adversarial Networks](https://link.springer.com/chapter/10.1007/978-3-030-00536-8_1); Hoo-Chang Shin, Neil A. Tenenholtz, Jameson K. Rogers, Christopher G. SchwarzMatthew L. Senjem, Jeffrey L. Gunter, Katherine P. Andriole, Mark Michalski; Data diversity is critical to success when training deep learning models. Medical imaging data sets are often imbalanced as pathologic findings are generally rare, which introduces significant challenges when training deep learning models. In this work, we propose a method to generate synthetic abnormal MRI images with brain tumors by training a generative adversarial network using two publicly available data sets of brain MRI. We demonstrate two unique benefits that the synthetic images provide. First, we illustrate improved performance on tumor segmentation by leveraging the synthetic images as a form of data augmentation. Second, we demonstrate the value of generative models as an anonymization tool, achieving comparable tumor segmentation results when trained on the synthetic data versus when trained on real subject data. Together, these results offer a potential solution to two of the largest challenges facing machine learning in medical imaging, namely the small incidence of pathological findings, and the restrictions around sharing of patient data.


## 2017


# Repositories

## Computer vision

#### - [albumentations](https://github.com/albu/albumentations) is a python library with a set of useful, large and diverse data augmentation methods. It offers over 30 different types of augmentations, easy and ready to use. Moreover, as the authors prove, the library is faster than other libraries on most of the transformations. 

Example jupyter notebooks:
* [All in one showcase notebook](https://github.com/albu/albumentations/blob/master/notebooks/showcase.ipynb)
* [Classification](https://github.com/albu/albumentations/blob/master/notebooks/example.ipynb),
* [Object detection](https://github.com/albu/albumentations/blob/master/notebooks/example_bboxes.ipynb),  [image segmentation](https://github.com/albu/albumentations/blob/master/notebooks/example_kaggle_salt.ipynb) and  [keypoints](https://github.com/albu/albumentations/blob/master/notebooks/example_keypoints.ipynb)
* Others - [Weather transforms ](https://github.com/albu/albumentations/blob/master/notebooks/example_weather_transforms.ipynb),
 [Serialization](https://github.com/albu/albumentations/blob/master/notebooks/serialization.ipynb),
 [Replay/Deterministic mode](https://github.com/albu/albumentations/blob/master/notebooks/replay.ipynb),  [Non-8-bit images](https://github.com/albu/albumentations/blob/master/notebooks/example_16_bit_tiff.ipynb)

Example tranformations:
![albumentations examples](https://s3.amazonaws.com/assertpub/image/1809.06839v1/image-002-000.png)

#### - [imgaug](https://github.com/aleju/imgaug) - is another very useful and widely used python library. As authors describe: *it helps you with augmenting images for your machine learning projects. It converts a set of input images into a new, much larger set of slightly altered images.* It offers many augmentation techniques such as affine transformations, perspective transformations, contrast changes, gaussian noise, dropout of regions, hue/saturation changes, cropping/padding, blurring.

Example jupyter notebooks:
* [Load and Augment an Image](https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/A01%20-%20Load%20and%20Augment%20an%20Image.ipynb)
* [Multicore Augmentation](https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/A03%20-%20Multicore%20Augmentation.ipynb)
 * Augment and work with: [Keypoints/Landmarks](https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B01%20-%20Augment%20Keypoints.ipynb),
    [Bounding Boxes](https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B02%20-%20Augment%20Bounding%20Boxes.ipynb),
    [Polygons](https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B03%20-%20Augment%20Polygons.ipynb),
    [Line Strings](https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B06%20-%20Augment%20Line%20Strings.ipynb),
    [Heatmaps](https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B04%20-%20Augment%20Heatmaps.ipynb),
    [Segmentation Maps](https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B05%20-%20Augment%20Segmentation%20Maps.ipynb) 

Example tranformations:
![imgaug examples](https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/examples_grid.jpg)

#### - [UDA](https://github.com/google-research/uda) - a simple data augmentation tool for image files, intended for use with machine learning data sets. The tool scans a directory containing image files, and generates new images by performing a specified set of augmentation operations on each file that it finds. This process multiplies the number of training examples that can be used when developing a neural network, and should significantly improve the resulting network's performance, particularly when the number of training examples is relatively small.
The details are avaible here: ![UNSUPERVISED DATA AUGMENTATION FOR CONSISTENCY TRAINING](https://arxiv.org/pdf/1904.12848.pdf)

#### - [Data augmentation for object detection](https://github.com/Paperspace/DataAugmentationForObjectDetection) - Repository contains a code for the paper [space tutorial series on adapting data augmentation methods for object detection tasks](https://blog.paperspace.com/data-augmentation-for-bounding-boxes/). They support a lot of data augmentations, like Horizontal Flipping, Scaling, Translation, Rotation, Shearing, Resizing.
![Data augmentation for object detection - exmpale](https://blog.paperspace.com/content/images/2018/09/vanila_aug.jpg)

#### - [Image augmentor](https://github.com/codebox/image_augmentor) - This is a simple python data augmentation tool for image files, intended for use with machine learning data sets. The tool scans a directory containing image files, and generates new images by performing a specified set of augmentation operations on each file that it finds. This process multiplies the number of training examples that can be used when developing a neural network, and should significantly improve the resulting network's performance, particularly when the number of training examples is relatively small.

#### - [torchsample](https://github.com/ncullen93/torchsample) - this python package provides High-Level Training, Data Augmentation, and Utilities for Pytorch. This toolbox provides data augmentation methods, regularizers and other utility functions. These transforms work directly on torch tensors:
* Compose()
* AddChannel()
* SwapDims()
* RangeNormalize()
* StdNormalize()
* Slice2D()
* RandomCrop()
* SpecialCrop()
* Pad()
* RandomFlip()


#### - [Random erasing](https://github.com/zhunzhong07/Random-Erasing) - The code is based on the paper: https://arxiv.org/abs/1708.04896. The Absract:

In this paper, we introduce Random Erasing, a new data augmentation method for training the convolutional neural network (CNN). In training, Random Erasing randomly selects a rectangle region in an image and erases its pixels with random values. In this process, training images with various levels of occlusion are generated, which reduces the risk of over-fitting and makes the model robust to occlusion. Random Erasing is parameter learning free, easy to implement, and can be integrated with most of the CNN-based recognition models. Albeit simple, Random Erasing is complementary to commonly used data augmentation techniques such as random cropping and flipping, and yields consistent improvement over strong baselines in image classification, object detection and person re-identification. Code is available at: this https URL.

![Example of random erasing](https://github.com/zhunzhong07/Random-Erasing/raw/master/all_examples-page-001.jpg)

#### - [data augmentation in C++](https://github.com/takmin/DataAugmentation) - Simple image augmnetation program transform input images with rotation, slide, blur, and noise to create training data of image recognition.

#### - [Data augmentation with GANs](https://github.com/AntreasAntoniou/DAGAN) - This repository contain files with Generative Adversarial Network, which can be used to successfully augment the dataset. This is an implementation of DAGAN as described in https://arxiv.org/abs/1711.04340. The implementation provides data loaders, model builders, model trainers, and synthetic data generators for the Omniglot and VGG-Face datasets.


## Natural Language Processing

#### - [Contextual data augmentation](https://github.com/pfnet-research/contextual_augmentation) - Contextual augmentation is a domain-independent data augmentation for text classification tasks. Texts in supervised dataset are augmented by replacing words with other words which are predicted by a label-conditioned bi-directional language model. 
This repository contains a collection of scripts for an experiment of [Contextual Augmentation](https://arxiv.org/pdf/1805.06201.pdf).

![example contextual data augmentation](https://i.imgur.com/JOyKkVt.png) 

#### - [nlpaug](https://github.com/makcedward/nlpaug) - This python library helps you with augmenting nlp for your machine learning projects. Visit this introduction to understand about [Data Augmentation in NLP](https://towardsdatascience.com/data-augmentation-in-nlp-2801a34dfc28). `Augmenter` is the basic element of augmentation while `Flow` is a pipeline to orchestra multi augmenter together.

Features:
 *   Generate synthetic data for improving model performance without manual effort
 *   Simple, easy-to-use and lightweight library. Augment data in 3 lines of code
 *   Plug and play to any neural network frameworks (e.g. PyTorch, TensorFlow)
 *   Support textual and audio input

![example textual augmentations](https://github.com/makcedward/nlpaug/raw/master/res/textual_example.png)
![Example audio augmentations](https://github.com/makcedward/nlpaug/raw/master/res/audio_example.png)


#### - [EDA NLP](https://github.com/jasonwei20/eda_nlp) - **EDA** is an **e**asy **d**ata **a**ugmentation techniques for boosting performance on text classification tasks. These are a generalized set of data augmentation techniques that are easy to implement and have shown improvements on five NLP classification tasks, with substantial improvements on datasets of size `N < 500`. While other techniques require you to train a language model on an external dataset just to get a small boost, we found that simple text editing operations using EDA result in good performance gains. Given a sentence in the training set, we perform the following operations:

- **Synonym Replacement (SR):** Randomly choose *n* words from the sentence that are not stop words. Replace each of these words with one of its synonyms chosen at random.
- **Random Insertion (RI):** Find a random synonym of a random word in the sentence that is not a stop word. Insert that synonym into a random position in the sentence. Do this *n* times.
- **Random Swap (RS):** Randomly choose two words in the sentence and swap their positions. Do this *n* times.
- **Random Deletion (RD):** For each word in the sentence, randomly remove it with probability *p*.


## Audio
#### - [SpecAugment with Pytorch](https://github.com/zcaceres/spec_augment) - (https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html) is a state of the art data augmentation approach for speech recognition. It supports augmentations such as time wrap, time mask, frequency mask or all above combined.

![time warp aug](https://github.com/zcaceres/spec_augment/raw/master/img/timewarp.png)

![time mask aug](https://github.com/zcaceres/spec_augment/raw/master/img/timemask.png)


#### - [Audiomentations](https://github.com/iver56/audiomentations) - A Python library for audio data augmentation. Inspired by albumentations. Useful for machine learning. It allows to use effects such as: Compose, AddGaussianNoise, TimeStretch, PitchShift and Shift.

#### - [MUDA](https://github.com/bmcfee/muda) - A library for Musical Data Augmentation. Muda package implements annotation-aware musical data augmentation, as described in the muda paper.
The goal of this package is to make it easy for practitioners to consistently apply perturbations to annotated music data for the purpose of fitting statistical models.
