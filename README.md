# Siamese-Triplet-Loss
# Single-Morphing Attack Detection Using Few-Shot Learning and Triplet-Loss.
This is a joint effort of Juan E. Tapia, Daniel Schulz and Christoph Busch.

This work proposes to use a triplet-loss function to estimate the difficulties of each morphing tool and examine different pre-trained networks as a general framework solution. The experiments are conducted with four datasets: FERET, FRGCv2, open-source AMSL Morph, and the state-of-the-art "Synthetic-Face-Morphing-Attack-Detection-Development-dataset" (SMDD). Each one presents subsets of different morphing tools and conditions.

As an essential contribution, an FSL includes only a small number of examples from a new unknown dataset to guide the training process and increase the method's performance. These examples allow us to assign the different morphing tools and attacks correctly.

# Description Method

 General representation of our modified Siamese network for Morphing attack detection. The images are cropped before being input into the network. Xa, Xb and Xn represent the anchor, positive and negative images, respectively. The f(Xa), f(Xb), and f(Xn) represent the embedding space for each image after feature extraction. The three possible distances for negative examples (easy, hard, semi-hard) are included as references.

<img width="761" alt="framework_sia7" src="https://github.com/user-attachments/assets/718d68b9-9df3-4805-b8e6-00f78833b806" />

# Implementation
- Download the files to your own folder and enviroment based on python 3.0 and tensforflow 2.10.0. (tf_2.10.0)
- Explore siameses netwrok.py to define level of data-augmentation, parameters (epoch, lr and others) and loss.
- Three loss implementation are available: constractive loss, Triplet_semi_hard_loss and Triplet_hard_loss
- To run the algorithm it is neccesary to create three txt files (train.txt, val.txt and test.txt). 
- The structure should be: name_image label;
- The label must be 0 for attack or 1 for bona fide.

```
python siamese_network.py

```
# Dataset

All the datasets must be requested directly to the sources described in the paper:
- Face Research Lab London
- Synthetic MAD evaluation benchmark datasets
- FRGC
- FERET

# Citation:
```
TBA

```
# License
This work and the methods proposed are only for research purposes, and the images are generated by chance. Any implementation or commercial use modification must be analysed separately for each case to the email: juan.tapia-farias@h-da.de.
