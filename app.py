"""
app.py — Streamlit Dashboard: Optimizer Comparison
Run with: streamlit run app.py
"""

import streamlit as st
import json
import os
import time
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Optimizer Comparison Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
        --bg: #0a0e1a;
        --surface: #111827;
        --surface2: #1a2235;
        --accent: #8b5cf6;
        --accent2: #818cf8;
        --green: #10b981;
        --amber: #f59e0b;
        --rose: #f43f5e;
        --cyan: #06b6d4;
        --text: #f8fafc;
        --muted: #cbd5f5;
        --border: rgba(99,102,241,0.2);
    }
    label, .css-1cpxqw2 {
        color: #e2e8f0 !important;
    }
    label, .stMarkdown, .stText {
        color: #e5e7eb !important;
    }        
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
        background-color: var(--bg);
        color: var(--text);
    }

    .stApp { background-color: var(--bg); }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #818cf8, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.5rem 0;
    }
    .hero p {
        color: var(--muted);
        font-size: 1rem;
        margin: 0;
    }

    /* Metric cards */
    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: var(--accent2); }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--text);
    }
    .metric-sub {
        font-size: 0.8rem;
        color: var(--muted);
        margin-top: 0.2rem;
        color: #cbd5f5;
    }

    /* Analysis box */
    .analysis-box {
        background: var(--surface);
        border-left: 3px solid var(--accent);
        border-radius: 0 12px 12px 0;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .analysis-box h4 {
        color: var(--accent2);
        margin: 0 0 0.75rem 0;
        font-size: 1rem;
        font-weight: 600;
    }
    .analysis-box p {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.7;
        margin: 0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;   /* brighter text */
    }
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, var(--accent), #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.2s !important;
        width: 100%;
    }
    .stButton button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
    }

    /* Selectbox */
    .stSelectbox [data-baseweb="select"] {
        background-color: var(--surface2) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
    }

    /* Table */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* Section header */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--accent2);
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
        letter-spacing: 0.03em;
        color: #a5b4fc;
    }
    .section-header {
        color: #c7d2fe;
    }
    /* Badge */
    .badge {
        display: inline-block;
        background: rgba(99,102,241,0.15);
        color: var(--accent2);
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 6px;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .stCodeBlock, pre, code {
        background-color: #111827 !important;   /* dark background */
        color: #e2e8f0 !important;              /* bright text */
        border-radius: 10px !important;
    }
    pre span {
        color: #e2e8f0 !important;
    }
    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

OPTIMIZER_COLORS = {
    "Adam":    "#6366f1",
    "SGD":     "#10b981",
    "RMSprop": "#f59e0b",
    "Adagrad": "#f43f5e",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.6)",
    font=dict(family="Space Grotesk", color="#94a3b8"),
    legend=dict(
        bgcolor="rgba(17,24,39,0.8)",
        bordercolor="rgba(99,102,241,0.2)",
        borderwidth=1,
    ),
    xaxis=dict(gridcolor="rgba(99,102,241,0.1)", linecolor="rgba(99,102,241,0.2)"),
    yaxis=dict(gridcolor="rgba(99,102,241,0.1)", linecolor="rgba(99,102,241,0.2)"),
    margin=dict(l=40, r=20, t=40, b=40),
)


@st.cache_data
def load_results():
    """Load pre-computed results from JSON (cached for performance)."""
    path = "results/comparison_results.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def train_and_save():
    """Import train.py logic and run the full comparison."""
    from train import run_comparison
    with st.spinner("Training all 4 optimizers… this takes ~30 seconds ☕"):
        results = run_comparison()
    st.cache_data.clear()   # refresh cache so load_results picks up new file
    return results


def metric_card(label, value, sub=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# CHARTS
# ════════════════════════════════════════════════════════════════════════════

def plot_accuracy(results):
    fig = go.Figure()
    for name, data in results.items():
        epochs = list(range(1, len(data["val_acc_history"]) + 1))
        fig.add_trace(go.Scatter(
            x=epochs, y=[v * 100 for v in data["val_acc_history"]],
            name=name, mode="lines",
            line=dict(color=OPTIMIZER_COLORS[name], width=2.5),
            hovertemplate=f"<b>{name}</b><br>Epoch %{{x}}<br>Accuracy: %{{y:.1f}}%<extra></extra>"
        ))
    fig.update_layout(
        title=dict(text="Validation Accuracy vs Epochs", font=dict(size=15, color="#e2e8f0")),
        xaxis_title="Epoch", yaxis_title="Accuracy (%)",
        **PLOTLY_LAYOUT
    )
    return fig


def plot_loss(results):
    fig = go.Figure()
    for name, data in results.items():
        epochs = list(range(1, len(data["val_loss_history"]) + 1))
        fig.add_trace(go.Scatter(
            x=epochs, y=data["val_loss_history"],
            name=name, mode="lines",
            line=dict(color=OPTIMIZER_COLORS[name], width=2.5),
            hovertemplate=f"<b>{name}</b><br>Epoch %{{x}}<br>Loss: %{{y:.4f}}<extra></extra>"
        ))
    fig.update_layout(
        title=dict(text="Validation Loss vs Epochs", font=dict(size=15, color="#e2e8f0")),
        xaxis_title="Epoch", yaxis_title="Loss",
        **PLOTLY_LAYOUT
    )
    return fig


def plot_bar_comparison(results):
    names = list(results.keys())
    accs  = [results[n]["accuracy"] for n in names]
    colors = [OPTIMIZER_COLORS[n] for n in names]

    fig = go.Figure(go.Bar(
        x=names, y=accs,
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.1)", width=1)),
        text=[f"{a}%" for a in accs],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=13, family="JetBrains Mono"),
        hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.2f}%<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text="Final Test Accuracy by Optimizer", font=dict(size=15, color="#e2e8f0")),
        
        **PLOTLY_LAYOUT
    )
    yaxis=dict(range=[min(accs) - 2, 100])
    return fig


def plot_single_optimizer(results, name):
    data   = results[name]
    epochs = list(range(1, len(data["train_acc_history"]) + 1))
    color  = OPTIMIZER_COLORS[name]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=[v * 100 for v in data["train_acc_history"]],
        name="Train Accuracy", mode="lines",
        line=dict(color=color, width=2.5, dash="solid")
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=[v * 100 for v in data["val_acc_history"]],
        name="Val Accuracy", mode="lines",
        line=dict(color=color, width=2, dash="dot"),
        fill="tonexty", fillcolor=f"rgba{tuple(int(color[i:i+2],16) for i in (1,3,5)) + (0.07,)}"
    ))
    fig.update_layout(
        title=dict(text=f"{name}: Train vs Val Accuracy", font=dict(size=15, color="#e2e8f0")),
        xaxis_title="Epoch", yaxis_title="Accuracy (%)",
        **PLOTLY_LAYOUT
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS CONTENT
# ════════════════════════════════════════════════════════════════════════════

ANALYSIS = {
    "why_adam": """
Adam (Adaptive Moment Estimation) combines the best of two worlds: momentum (which accelerates learning in the right direction) 
and adaptive learning rates (which gives each parameter its own dynamic step size). This makes it robust across 
diverse architectures and datasets, requiring minimal hyperparameter tuning — a huge practical advantage.
""",
    "when_others": """
• <b>SGD with Momentum</b>: Can outperform Adam on large-scale tasks (e.g., ResNet on ImageNet) when 
  carefully tuned with a learning rate schedule. It often generalizes better due to "wider minima" convergence.<br><br>
• <b>RMSprop</b>: Excellent for recurrent neural networks (RNNs, LSTMs) and non-stationary problems. 
  Historically the optimizer of choice for RNNs before Adam was widely adopted.<br><br>
• <b>Adagrad</b>: Performs well with sparse data and NLP tasks (e.g., text classification) where infrequent 
  features should receive larger updates. Its diminishing learning rate is a limitation for deep networks.
""",
}


def render_analysis(results):
    best = max(results, key=lambda n: results[n]["accuracy"])
    best_acc = results[best]["accuracy"]

    st.markdown('<div class="section-header">📊 Analysis & Findings</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="analysis-box">
        <h4>🏆 Best Performing Optimizer</h4>
        <p><b style="color:#818cf8">{best}</b> achieved the highest test accuracy of <b style="color:#10b981">{best_acc}%</b> 
        on the Breast Cancer Wisconsin dataset. This aligns with expectations given Adam's adaptive learning rate 
        mechanism, which converges efficiently even without extensive hyperparameter tuning.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="analysis-box">
        <h4>⚡ Why Adam is Generally Preferred</h4>
        <p>{ANALYSIS['why_adam'].strip()}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="analysis-box">
        <h4>🔄 When Other Optimizers Shine</h4>
        <p>{ANALYSIS['when_others'].strip()}</p>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════════════

def main():
    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <h1>⚡ Optimizer Comparison Dashboard</h1>
        <p>Breast Cancer Wisconsin · Neural Network · TensorFlow/Keras · Adam vs SGD vs RMSprop vs Adagrad</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        st.markdown("---")

        optimizer_choice = st.selectbox(
            "Select Optimizer to Inspect",
            ["Adam", "SGD", "RMSprop", "Adagrad"],
            index=0,
            help="Choose an optimizer to see its individual training curves"
        )

        st.markdown("---")
        train_btn = st.button("🚀 Train All Optimizers", use_container_width=True)

        st.markdown("---")
        st.markdown("**Dataset:** Breast Cancer Wisconsin")
        st.markdown("**Samples:** 569 · **Features:** 30")
        st.markdown("**Task:** Binary Classification")
        st.markdown("---")
        st.markdown("**Hyperparameters**")
        st.code("Epochs:     50\nBatch Size: 32\nLR:         0.001\nTest Split: 20%", language="text")
        
    # ── Train if button pressed ───────────────────────────────────────────
    if train_btn:
        results = train_and_save()
        st.success("✅ Training complete! Results saved.")
    else:
        results = load_results()

    # ── No results yet ────────────────────────────────────────────────────
    if results is None:
        st.info("👈 Click **Train All Optimizers** in the sidebar to get started!")
        st.markdown("""
        <div class="analysis-box">
            <h4>ℹ️ How It Works</h4>
            <p>
            1. Click <b>Train All Optimizers</b><br>
            2. The app trains a Neural Network 4 times — once per optimizer<br>
            3. Results are displayed here interactively<br>
            4. Use the dropdown to inspect individual optimizer curves
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Overview Metrics ─────────────────────────────────────────────────
    # Find best optimizer
    best = max(results, key=lambda x: results[x]['accuracy'])

    st.markdown("### 🏆 Best Optimizer")

    st.success(f"{best} performed the best with {results[best]['accuracy']}% accuracy")
    st.markdown('<div class="section-header">📈 Overview Metrics</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (name, data) in enumerate(results.items()):
        with cols[i]:
            metric_card(
                f"{name}",
                f"{data['accuracy']}%",
                f"Loss: {data['loss']} · {data['training_time']}s"
            )

    # ── All Optimizers Charts ────────────────────────────────────────────
    st.markdown('<div class="section-header">📉 Training Curves — All Optimizers</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_accuracy(results), use_container_width=True)
    with c2:
        st.plotly_chart(plot_loss(results), use_container_width=True)

    # ── Bar Chart ────────────────────────────────────────────────────────
    st.plotly_chart(plot_bar_comparison(results), use_container_width=True)
    
    # ── Individual Optimizer Drill-Down ──────────────────────────────────
    st.markdown(f'<div class="section-header">🔍 Deep Dive: {optimizer_choice}</div>', unsafe_allow_html=True)

    d = results[optimizer_choice]
    col1, col2, col3, col4 = st.columns(4)
    with col1: metric_card("Test Accuracy",    f"{d['accuracy']}%")
    with col2: metric_card("Final Loss",       f"{d['loss']}")
    with col3: metric_card("Training Time",    f"{d['training_time']}s")
    with col4: metric_card("Convergence Epoch",f"#{d['convergence_epoch']}")

    st.plotly_chart(plot_single_optimizer(results, optimizer_choice), use_container_width=True)

    # ── Comparison Table ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Final Comparison Table</div>', unsafe_allow_html=True)

    df = pd.DataFrame([
        {
            "Optimizer":         name,
            "Accuracy (%)":      data["accuracy"],
            "Final Loss":        data["loss"],
            "Training Time (s)": data["training_time"],
            "Convergence Epoch": data["convergence_epoch"],
        }
        for name, data in results.items()
    ]).set_index("Optimizer")

    best_name = df["Accuracy (%)"].idxmax()

    def highlight_best(row):
        return ["background-color: rgba(99,102,241,0.15); font-weight:600;" if row.name == best_name else "" for _ in row]

    styled = df.style.apply(highlight_best, axis=1).format({
        "Accuracy (%)": "{:.2f}",
        "Final Loss":   "{:.4f}",
        "Training Time (s)": "{:.2f}",
    })
    st.dataframe(styled, use_container_width=True)

    # ── Analysis ─────────────────────────────────────────────────────────
    render_analysis(results)
    
    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:#475569;font-size:0.8rem;'>"
        "ML Coursework · Breast Cancer Wisconsin · TensorFlow/Keras · Streamlit"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
