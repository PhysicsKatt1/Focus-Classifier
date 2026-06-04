# Machine Learning for Automated Electron Microscopy Correction and Instrument Health Monitoring

## Overview

This repository contains machine learning research focused on developing intelligent systems for automated instrument monitoring, aberration correction, and image analysis in advanced electron microscopy environments.

The project investigates how deep learning and computer vision techniques can be used to improve the reliability, throughput, and automation of semiconductor manufacturing workflows involving Scanning Electron Microscopy (SEM) and Focused Ion Beam (FIB) systems.

By leveraging large-scale microscopy datasets, this work aims to reduce manual intervention, improve instrument performance, and enable real-time corrective actions during imaging and manufacturing processes.

---

## Research Objectives

### Instrument Health Monitoring

Electron microscopy systems can experience performance degradation over time due to changes in instrument alignment, voltage drift, contamination, or hardware instability.

This project develops machine learning models capable of:

* Detecting early indicators of instrument degradation
* Monitoring system health in real time
* Identifying abnormal operating conditions
* Supporting predictive maintenance workflows

The goal is to improve reliability while reducing downtime and unnecessary service interventions.

### Automated Aberration Correction

Microscope imaging quality is strongly influenced by optical aberrations such as:

* Astigmatism
* Defocus
* Beam misalignment

This research explores the use of deep learning models to:

* Identify aberration signatures directly from image data
* Predict required correction parameters
* Automate Focused Ion Beam alignment procedures
* Reduce operator workload and increase manufacturing throughput

---

## Methodology

### Data Processing

Large microscopy image datasets are collected from semiconductor manufacturing environments and preprocessed through:

* Image normalization
* Data augmentation
* Feature extraction
* Dataset balancing
* Quality validation

### Machine Learning Pipeline

The workflow includes:

1. Data acquisition
2. Data preprocessing
3. Model training
4. Hyperparameter optimization
5. Validation and testing
6. Performance monitoring

### Deep Learning Approaches

Models investigated include:

* Convolutional Neural Networks (CNNs)
* Binary classification networks
* Transfer learning architectures
* Custom image analysis pipelines

Primary objectives include:

* Classification of healthy vs degraded system states
* Aberration detection
* Correction parameter prediction
* Robust performance under varying imaging conditions

---

## Technologies Used

### Programming Languages

* Python

### Machine Learning Frameworks

* TensorFlow
* PyTorch
* Keras
* Scikit-learn

### Scientific Computing

* NumPy
* SciPy
* Pandas

### Computer Vision

* OpenCV

### Data Visualization

* Matplotlib

### Infrastructure

* Linux
* Git
* Kubernetes

---

## Key Outcomes

### Real-Time Health Monitoring

Developed generalized binary classification models capable of detecting microscope performance degradation and supporting automated monitoring workflows.

### Automated Alignment Correction

Designed machine learning workflows that correlate imaging artifacts with instrument control parameters, enabling automated correction of Focused Ion Beam alignment errors.

### Improved Manufacturing Efficiency

Contributed to efforts aimed at:

* Reducing manual intervention
* Increasing imaging consistency
* Improving instrument uptime
* Enhancing semiconductor manufacturing throughput

### Scalable AI Workflows

Built reusable machine learning pipelines capable of supporting future applications in scientific imaging and automated instrumentation.

---

## Research Areas

This work sits at the intersection of:

* Machine Learning
* Computer Vision
* Electron Microscopy
* Semiconductor Manufacturing
* Scientific Computing
* Instrumentation Automation
* Artificial Intelligence for Scientific Discovery

---

## Future Work

Potential future directions include:

* Vision Transformers (ViTs) for microscopy image analysis
* Self-supervised learning on unlabeled microscopy datasets
* Real-time deployment on production instruments
* Reinforcement learning for automated instrument tuning
* Explainable AI methods for model interpretability
* Multimodal models combining image and instrument telemetry data

---

## Author

**Katherine Kaylegian-Starkey**

Machine Learning Researcher | Physicist | Systems Engineer

Focused on applying artificial intelligence to scientific imaging, advanced instrumentation, and semiconductor manufacturing.

LinkedIn: linkedin.com/in/physicskatt

GitHub: github.com/PhysicsKatt
