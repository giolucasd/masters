# MIA-3DCNN: COVID-19 DETECTION BASED ON A 3D CNN

## 1. Authors

* Igor Kenzo Ishikawa Oshiro Nakashima
* Giovanna Vendramini
* Helio Pedrini

## 2. Abstract
 Early and accurate diagnosis of COVID-19 is essential to control the rapid spread of the pandemic and mitigate sequelae in the population.
 Current diagnostic methods, such as RT-PCR, are effective but require time to provide results and can quickly overwhelm clinics, requiring individual laboratory analysis.
 Automatic detection methods have the potential to significantly reduce diagnostic time.
 To this end, learning-based methods using lung imaging have been explored.
 Although they require specialized hardware, automatic evaluation methods can be performed simultaneously, making diagnosis faster.
 Convolutional neural networks have been widely used to detect pneumonia caused by COVID-19 in lung images.
 This work describes an architecture based on 3D convolutional neural networks for detecting COVID-19 in computed tomography images.
 Despite the challenging scenario present in the dataset, the results obtained with our architecture demonstrated to be quite promising.

 ## 3. Notes

 ## 3.1. I/O

* Input: 3D MRI image
* Output:
    * Covid detection (classification on positive or negative)
    * Severity classification (classification on 4 classes of severity)

### 3.2. Data preprocessing

Resizing with spline interpolation and data augmentation.

### 3.3. Architecture

Proposed architecture is a 3D CNN composed of two main stages, one composed of 3D convolutional blocks, and one composed of fully connected layers.

Layers and parameters were decided upon experimentation.

### 3.4. Questions

* How where the experiments for architecture definition done? Are they intuitivelly approachable? Does literature supports most of the decisions?

* Can data augmentation help prevent overfitting? If yes, why? Does it have a realtion with gaussian blur and noise?

## 4. Conslusions

The proposed architecture surpass baseline results for both tasks.