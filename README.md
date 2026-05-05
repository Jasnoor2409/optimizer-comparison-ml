# ⚡ Optimizer Comparison Dashboard

An interactive machine learning project that compares the performance of different optimization algorithms — **Adam, SGD, RMSprop, and Adagrad** — on a neural network classification task using the **Breast Cancer Wisconsin Dataset**.

The project includes a **Streamlit dashboard** to visualize results and analyze optimizer behavior in real time.

---

## 📌 Features

* Compare multiple optimizers on the same neural network
* Interactive dashboard built with Streamlit
* Visualization of:

  * Accuracy
  * Loss
  * Training time
  * Convergence speed
* Clean and modular code structure
* Reproducible experiments

---

## 📁 Project Structure

```
optimizer_comparison/
├── train.py              # Model training and evaluation logic
├── app.py                # Streamlit dashboard
├── requirements.txt      # Dependencies
├── results/              # Generated results (JSON)
│   └── comparison_results.json
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```
git clone https://github.com/your-username/optimizer-comparison-ml.git
cd optimizer-comparison-ml
```

---

### 2. Create virtual environment (recommended)

**Windows:**

```
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**

```
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

### 4. Run the project

#### Option A — Run training script

```
python train.py
```

This generates:

```
results/comparison_results.json
```

---

#### Option B — Launch dashboard

```
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

Use the sidebar button **"Train All Optimizers"** to run experiments.

---

## 🧠 Model Architecture

* Input Layer: 30 features
* Dense Layer: 64 neurons (ReLU)
* Dropout: 0.2
* Dense Layer: 32 neurons (ReLU)
* Output Layer: 1 neuron (Sigmoid)

**Loss Function:** Binary Crossentropy
**Evaluation Metric:** Accuracy

---

## ⚙️ Training Configuration

| Parameter        | Value |
| ---------------- | ----- |
| Epochs           | 50    |
| Batch Size       | 32    |
| Learning Rate    | 0.001 |
| Train/Test Split | 80/20 |
| Random Seed      | 42    |

---

## 📊 Optimizers Compared

| Optimizer | Description                                                 |
| --------- | ----------------------------------------------------------- |
| Adam      | Adaptive learning rate with momentum                        |
| SGD       | Stochastic Gradient Descent with momentum                   |
| RMSprop   | Adaptive learning using moving average of squared gradients |
| Adagrad   | Adaptive learning for sparse features                       |

---

## 📈 Evaluation Metrics

* **Accuracy** — Final test performance
* **Loss** — Validation loss
* **Training Time** — Execution duration
* **Convergence Speed** — Epoch at which model stabilizes

---

## 🎯 Key Objective

To analyze how different optimization algorithms affect:

* Training speed
* Model convergence
* Final performance

All experiments are conducted using the **same model architecture** to ensure a fair comparison.

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* Scikit-learn
* Streamlit
* Plotly
* Pandas / NumPy

---

## 📌 Future Improvements

* Add more optimizers (AdamW, Nadam)
* Hyperparameter tuning
* Support for multiple datasets
* Save and compare experiment runs

---

This project demonstrates practical implementation of optimization algorithms and their impact on neural network performance.
