# Description

Trains a convolutional neural network (CNN) on custom (small) datasets using a novel, orientation-preserving feature space augmentation technique. Performs data augmentation (rotation, translation, shearing, etc.) on elements of a feature space dictionary generated at every training epoch in order to increase the dataset size and thereby improve model accuracy while also maintaining clarity and efficacy of generated feature maps, maximum activation visualizations, and heatmaps (such as those from Grad-CAMs).

Default CNN architecture:
![network](https://user-images.githubusercontent.com/33817654/100400521-b38bf700-300b-11eb-98df-ca7084a258e1.jpg)


# Installation

```
pip install fsaug
```
