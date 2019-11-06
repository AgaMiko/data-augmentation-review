# Data augmentation
List of useful data augmentation resources. You will find here some not common techniques, libraries, links to github repos, papers and others.

* [Introduction](https://github.com/AgaMiko/data-augmentation/blob/master/README.md#Introduction)
* [Common techniques](https://github.com/AgaMiko/data-augmentation/blob/master/README.md#Common-techniques)
* [Papers](https://github.com/AgaMiko/data-augmentation/blob/master/README.md#Papers)
* [Repositories](https://github.com/AgaMiko/data-augmentation/blob/master/README.md#Repositories)
* [Libraries](https://github.com/AgaMiko/data-augmentation/blob/master/README.md#Libraries)


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


## Neural-based transformations

# Sound

# Papers

# Repositories

# Libraries
- [albumentations](https://github.com/albu/albumentations) is a python library with a set of useful, large and diverse data augmentation methods. It offers over 30 different types of augmentations, easy and ready to use. Moreover, as the authors prove, the library is faster than other libraries on most of the transformations. 

Example jupyter notebooks:
* [All in one showcase notebook](https://github.com/albu/albumentations/blob/master/notebooks/showcase.ipynb)
* [Classification](https://github.com/albu/albumentations/blob/master/notebooks/example.ipynb),
* [Object detection](https://github.com/albu/albumentations/blob/master/notebooks/example_bboxes.ipynb),  [image segmentation](https://github.com/albu/albumentations/blob/master/notebooks/example_kaggle_salt.ipynb) and  [keypoints](https://github.com/albu/albumentations/blob/master/notebooks/example_keypoints.ipynb)
* Others - [Weather transforms ](https://github.com/albu/albumentations/blob/master/notebooks/example_weather_transforms.ipynb),
 [Serialization](https://github.com/albu/albumentations/blob/master/notebooks/serialization.ipynb),
 [Replay/Deterministic mode](https://github.com/albu/albumentations/blob/master/notebooks/replay.ipynb),  [Non-8-bit images](https://github.com/albu/albumentations/blob/master/notebooks/example_16_bit_tiff.ipynb)

Example tranformations:
![albumentations examples](https://s3.amazonaws.com/assertpub/image/1809.06839v1/image-002-000.png)

- [imgaug](https://github.com/aleju/imgaug) - is another very useful and widely used python library. As authors describe: *it helps you with augmenting images for your machine learning projects. It converts a set of input images into a new, much larger set of slightly altered images.* It offers many augmentation techniques such as affine transformations, perspective transformations, contrast changes, gaussian noise, dropout of regions, hue/saturation changes, cropping/padding, blurring.

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


