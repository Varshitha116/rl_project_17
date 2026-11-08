# Project 17: Mini NAS Controller — REINFORCE Policy Gradient

> **Course**: Unit 5 — 25 Real-World RL Homework Projects  
> **Category**: RL for Hyperparameter Optimization and Neural Architecture Search  
> **Algorithm**: REINFORCE with Exponential Moving Average Baseline  
> **Author**: Varshitha

---

## 📋 Overview

Designing optimal neural network architectures is time-consuming and requires expert knowledge. This project trains an **RNN-based controller** using the **REINFORCE** policy gradient algorithm to automatically propose compact CNN architectures for CIFAR-10, replacing manual architecture design with learned search.

The controller autoregressively generates 12 architectural decisions (filters, kernel size, activation, residual connection for each of 3 layers), and each proposed architecture is evaluated by training a child CNN for a few epochs. The validation accuracy serves as the reward signal to improve the controller's policy.

---

## 🧠 MDP Formulation

| Component | Description |
|-----------|-------------|
| **State** | Implicit — autoregressive LSTM hidden state carries context from previous decisions |
| **Action** | 12 sequential discrete decisions per architecture (4 choices × 3 layers) |
| **Reward** | `val_acc × 100 − 0.05 × (params / 10000)` — accuracy with parameter efficiency penalty |
| **Episode** | One full architecture proposal → child training → validation evaluation |

### Search Space (per layer × 3 layers)

| Decision | Options |
|----------|---------|
| Filters | {16, 32, 64} |
| Kernel Size | {3, 5} |
| Activation | {ReLU, LeakyReLU, GELU} |
| Residual Connection | {Yes, No} |

**Total search space**: (3 × 2 × 3 × 2)³ = **46,656 unique architectures**

---

## 🏗 Architecture

### RNN Controller
- **LSTMCell**-based autoregressive generator (hidden_dim = 64)
- 12 linear output heads, one per architectural decision
- Each sampled action is embedded and fed as input to the next step
- Trained with REINFORCE + entropy bonus (coefficient = 0.01)

### Dynamic Child CNN
- Modular `ConvBlock` layers: Conv2d → BatchNorm → Activation → optional Residual → MaxPool
- Global Average Pooling + Linear classifier
- Architecture dynamically constructed from controller's sampled tokens
- Each candidate trained for 3 epochs with Adam (lr = 0.003)

### Training Details
- **Baseline**: Exponential moving average (decay = 0.85) for variance reduction
- **Gradient clipping**: Norm 1.0 on controller parameters
- **Dataset**: CIFAR-10 subset (3000 train, 1500 val) for rapid evaluation

---

## 📊 Baselines Compared

| Method | Description |
|--------|-------------|
| **Random Architecture Search** | Uniformly samples architectures from the same search space |
| **REINFORCE Controller** *(ours)* | Learned RNN policy that proposes increasingly better architectures |

The learning curves plot shows:
- **Left panel**: Best-found architecture accuracy over search budget
- **Right panel**: Smoothed reward signal progression (moving average)

---

## 🗂 Repository Structure

```
├── README.md                            # This file
├── 17_problem_formulation.pdf           # One-page MDP formulation (state/action/reward)
├── 17_implementation_reinforce.py       # REINFORCE NAS controller implementation
├── 17_learning_curves.png              # Comparative learning curves plot
├── 17_presentation_5slides.pptx        # 5-slide presentation
└── requirements.txt                    # Python dependencies
```

---

## 🚀 How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the NAS Search
```bash
python 17_implementation_reinforce.py --iterations 20 --child_epochs 3 --subset 3000 --out_dir 17
```

### CLI Arguments
| Flag | Default | Description |
|------|---------|-------------|
| `--iterations` | 20 | Total architecture search iterations |
| `--child_epochs` | 3 | Epochs to train each proposed child model |
| `--subset` | 3000 | CIFAR-10 training subset size for rapid evaluation |
| `--out_dir` | 17 | Output directory for plots, checkpoints, and results |

### Output Files
- `controller_checkpoint.pt` — Trained RNN controller weights
- `results_summary.json` — Best architecture config and accuracy metrics
- `learning_curves.png` — Two-panel comparative plot

---

## 📦 Deliverables Checklist

| # | Deliverable | File | Status |
|---|-------------|------|--------|
| 1 | Problem Formulation (1-page PDF) | `17_problem_formulation.pdf` | ✅ |
| 2 | Implementation (REINFORCE algorithm) | `17_implementation_reinforce.py` | ✅ |
| 3 | Learning Curves (vs. Random NAS baseline) | `17_learning_curves.png` | ✅ |
| 4 | 5-Slide Presentation | `17_presentation_5slides.pptx` | ✅ |

---

## ⚙️ Requirements

- Python 3.8+
- PyTorch ≥ 1.10
- torchvision
- NumPy
- Matplotlib

---

## 📝 License

This project is submitted as coursework for academic purposes.
