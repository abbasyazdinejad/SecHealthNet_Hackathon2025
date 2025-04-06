# 🔐 SecHealthNet_Hackathon

### 🌍 Privacy-Preserving AI for Collaborative Healthcare

The **SecHealthNet_Hackathon** project demonstrates how five hospitals can collaboratively train an AI model using **Federated Learning** while preserving patient privacy through **Differential Privacy**.

We simulate this collaboration using the **PathMNIST** dataset from the **MedMNIST** collection, which includes over **100,000 labeled pathology slide images** across **9 tissue classes** (e.g., liver, kidney, lung). Each hospital receives a different data partition and trains a local **Convolutional Neural Network (CNN)** to perform a **multi-class medical image classification task**.

Using the **Opacus** library, we apply **Differential Privacy** to ensure that model gradients are securely shared with a central aggregator — never exposing raw patient data. After each federated round, encrypted model updates are aggregated into a **global model**, which becomes more accurate with each cycle.

This mirrors real-world scenarios where hospitals cannot centralize data but still need high-performing predictive models for tasks such as **risk scoring**, **diagnosis**, and **outcome prediction**. The code integrates key concepts from our solution architecture: **data decentralization**, **privacy preservation**, and **regulatory compliance**, making it a strong fit for AI in healthcare applications.

---

## 🧠 Project Features

- ✅ Federated Learning across 5 simulated hospitals
- 🔏 Differential Privacy using Opacus
- 🏥 Local model training with secure gradient sharing
- 🔗 Global model aggregation via parameter averaging
- 🧬 MedMNIST PathMNIST dataset for real medical imaging
- 📊 Final global accuracy evaluation
- 📷 Visual sample viewer for input images

---

## 🖼️ System Architecture

### 🔽 Federated Learning with Privacy

![Federated Learning](./h2.png)

---

### 🔐 SecureHealthNet Vision

![SecureHealthNet Overview](./h3.png)

---

## 📁 Repository Structure

