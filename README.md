# ⚡ Optimizer Comparison Dashboard
### ML Coursework — Breast Cancer Wisconsin Dataset

Compare Adam, SGD, RMSprop, and Adagrad on a Neural Network classification task using TensorFlow/Keras, visualized in an interactive Streamlit dashboard.

---

## 📁 Project Structure

```
optimizer_comparison/
├── train.py          ← Neural Network training + evaluation logic
├── app.py            ← Streamlit dashboard
├── requirements.txt  ← Python dependencies
├── results/          ← Auto-created after training
│   └── comparison_results.json
└── README.md
```

---

## 🚀 Setup & Run

### Step 1 — Clone / Download the project
Place all files in a folder called `optimizer_comparison/`.

### Step 2 — Create a virtual environment (recommended)
```bash
python -m venv venv

# Activate (Windows):
venv\Scripts\activate

# Activate (Mac/Linux):
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4A — Run standalone training (optional)
This trains all 4 optimizers and saves results to `results/comparison_results.json`.
```bash
python train.py
```

### Step 4B — Launch the Streamlit App
```bash
streamlit run app.py
```

Then open your browser at **http://localhost:8501**

In the app, click **"Train All Optimizers"** in the sidebar to start training.

---

## 🧠 Model Architecture

```
Input (30 features)
    ↓
Dense(64, ReLU)
    ↓
Dropout(0.2)
    ↓
Dense(32, ReLU)
    ↓
Dense(1, Sigmoid)  ← Binary output
```

**Loss:** Binary Crossentropy  
**Metrics:** Accuracy  

---

## ⚙️ Training Configuration

| Setting      | Value |
|-------------|-------|
| Epochs      | 50    |
| Batch Size  | 32    |
| Learning Rate | 0.001 |
| Train/Test Split | 80/20 |
| Random Seed | 42    |

---

## 📊 Optimizers Compared

| Optimizer | Configuration |
|-----------|--------------|
| Adam      | lr=0.001 (default betas) |
| SGD       | lr=0.001, momentum=0.9 |
| RMSprop   | lr=0.001 |
| Adagrad   | lr=0.001 |

---

## 📈 Evaluation Metrics

- **Accuracy** — Final test set accuracy
- **Loss** — Final validation loss
- **Training Time** — Wall-clock seconds
- **Convergence Epoch** — First epoch where val_loss < 50% of initial loss

---

## 🗣️ Viva Quick Notes

**Q: Why is Adam generally preferred?**
Adam combines momentum + adaptive learning rates per parameter. It requires minimal tuning and converges fast across diverse tasks.

**Q: When would you use SGD?**
SGD with momentum can achieve better generalization on large-scale vision tasks (ResNet, VGG) with careful LR scheduling.

**Q: When is Adagrad useful?**
Sparse data / NLP tasks where infrequent features need larger gradient updates.

**Q: Why same architecture for all optimizers?**
Scientific fairness — we isolate the optimizer as the only variable.

**Q: What is convergence speed?**
The epoch at which validation loss drops below 50% of its starting value — a proxy for how quickly the model learns.
