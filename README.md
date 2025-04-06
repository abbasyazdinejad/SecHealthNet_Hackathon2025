# 🔐 SecHealthNet_Hackathon2025

### 🌍 Privacy-Preserving AI for Collaborative Healthcare

The **SecHealthNet_Hackathon** project demonstrates how five hospitals can collaboratively train an AI model using **Federated Learning** while preserving patient privacy through **Differential Privacy**.

We simulate this collaboration using the **PathMNIST** dataset from the **MedMNIST** collection, which includes over **100,000 labeled pathology slide images** across **9 tissue classes** (e.g., liver, kidney, lung). Each hospital receives a different data partition and trains a local **Convolutional Neural Network (CNN)** to perform a **multi-class medical image classification task**.

We will apply **Privacy Techniques** to ensure that model gradients are securely shared with a central aggregator — never exposing raw patient data. After each federated round, encrypted model updates are aggregated into a **global model**, which becomes more accurate with each cycle.

This mirrors real-world scenarios where hospitals cannot centralize data but still need high-performing predictive models for tasks such as **risk scoring**, **diagnosis**, and **outcome prediction**. The code integrates key concepts from our solution architecture: **data decentralization**, **privacy preservation**, and **regulatory compliance**, making it a strong fit for AI in healthcare applications.

![image](https://github.com/user-attachments/assets/75acd015-0c42-4238-833b-6a5e0e7d5390)

---

## 🧠 Project Features

- ✅ Federated Learning across 5 simulated hospitals
- 🔏 Privacy Techniques (DP, HE, AHE, etc.)
- 🏥 Local model training with secure gradient sharing
- 🔗 Global model aggregation via parameter averaging
- 🧬 MedMNIST PathMNIST dataset for real medical imaging
- 📊 Final global accuracy evaluation
- 📷 Visual sample viewer for input images

---

## 🖼️ System Architecture

### 🔽 Federated Learning with  Privacy Techniques

### 🔐 SecureHealthNet Vision

![image](https://github.com/user-attachments/assets/23d06bcc-d043-4832-9251-c2caf5a80d5a)

- In our simulation, we demonstrate how five hospitals can collaboratively train a medical AI model using federated learning without ever sharing patient data. We use the PathMNIST dataset from MedMNIST, which contains over 100,000 labeled pathology slide images categorized into nine tissue types, such as liver, lung, and kidney. Each hospital receives a subset of this dataset and trains a local Convolutional Neural Network (CNN) to perform a multi-class image classification task, predicting the correct tissue class for each image. To preserve patient privacy, we utilize Privacy by design, ensuring that model updates are obfuscated enough to prevent data leakage. After local training, only the model parameters are shared with a central aggregator, which combines them into a global model. This process simulates real-world hospital collaboration where data cannot leave the institution, yet accurate AI models can still be built collectively to support tasks like cancer detection, organ identification, and diagnostics.
---

## 📁 Repository Structure

