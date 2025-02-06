# IoTID20 Network Intrusion Detection

This project aims to develop an Intrusion Detection System (IDS) for IoT networks using SVM (Support Vector Machine) classifiers.

The IoTID20 dataset offers detailed network traffic data, making it a good fit for accurate intrusion detection. However, its high dimensionality can be a challenge, as it can slow down IoT systems that require fast, efficient models for real-time processing.

To address this, we use dimensionality reduction techniques like LDA (Linear Discriminant Analysis) and PCA (Principal Component Analysis) to find the right balance of features. This helps maintain high accuracy while reducing computational load and speeding up predictions.

We then fine-tune our SVM classifiers to find the best parameters for our model. This is done for two types of classification: binary classification (to determine whether a request is an attack or not) and multiclass classification (to classify different types of attacks).

Finally, we compare the performance of the various SVM models using different metrics and visualizations, to better understand their strengths and weaknesses, and find the most effective model.

