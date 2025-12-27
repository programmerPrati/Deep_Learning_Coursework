# Deep Learning Coursework: Architectural Implementation

This repository contains a series of projects focused on the design, implementation, and optimization of deep neural networks. 
The curriculum progressed from  distance-based classifiers to generative architectures.

**Key Skills:** Python, PyTorch/NumPy, Computer Vision, Backpropagation, Optimization (Adam/SGD), Generative Modeling.

---

## k-Nearest Neighbors (KNN)
Established a baseline for image classification using distance-based logic.
* **Implementation:** Developed L1 and L2 (Euclidean) distance metrics to classify images based on local pixel similarity.
* **Concepts:** Explored the "Curse of Dimensionality" and used **Cross-Validation** to optimize the hyperparameter $k$.

## 2-Layer Neural Network
Implemented the transition from linear models to non-linear function approximators.
* **Math:** Derived and coded the **Chain Rule** and **Backpropagation** from scratch using only NumPy.
* **Architecture:** Integrated a hidden layer with ReLU activation and a Softmax loss function, achieving significantly higher accuracy than linear baselines.



## Fully Connected Networks (FCN)
Developed a modular framework for building deep networks of arbitrary depth.
* **Modularity:** Built reusable layers for **Linear transforms, Dropout (regularization), and Batch Normalization**.
* **Optimization:** Implemented advanced update rules (SGD+Momentum) to accelerate convergence and overcome vanishing gradient problems.

## Convolutional Neural Networks (CNN)
Engineered architectures specifically designed for spatial hierarchies in image data.
* **Architecture:** Implemented **Convolutional layers** and **Max-Pooling** to achieve spatial invariance and feature extraction.
* **Technique:** Explored deep architectures (VGG/ResNet style) and utilized **Spatial Batch Normalization** to stabilize training for high-resolution image classification.



## Generative Adversarial Networks (GAN)
Transitioned from discriminative to generative modeling by training two networks in competition.
* **Adversarial Logic:** Implemented a Generator to produce synthetic images from latent noise and a Discriminator to distinguish real from fake data.
* **Training:** Optimized the **Minimax objective function**, balancing the training of both networks to produce realistic outputs and exploring the challenges of mode collapse.



---

### Implementation Details
* **Frameworks:** Projects were implemented primarily in **PyTorch** and **NumPy**.
* **Hardware:** Utilized GPU acceleration to handle the computational intensity of deep CNNs and GANs.
* **Best Practices:** Focused on vectorized code for performance and weight initialization strategies (Xavier/He) to ensure stable training.
