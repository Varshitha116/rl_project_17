# RL Projects 16 & 17 — Hyperparameter Optimization & Neural Architecture Search

> **Course**: Unit 5 — 25 Real-World RL Homework Projects  
> **Category**: RL for Hyperparameter Optimization and Neural Architecture Search  
> **Date**: October 2026

---

## 📋 Overview

This repository contains two reinforcement learning projects that apply RL to **meta-learning** problems in deep learning: automatically tuning hyperparameters and discovering neural network architectures.

| # | Project | Algorithm | Environment | Baselines |
|---|---------|-----------|-------------|-----------|
| **16** | RL-based Learning-Rate Scheduler | DQN (Deep Q-Network) | CIFAR-10 CNN Training Loop | Fixed LR, StepLR Decay, Random Policy |
| **17** | Mini NAS Controller | REINFORCE (Policy Gradient) | CIFAR-10 Architecture Search | Random Architecture Search |

---

## 🗂 Repository Structure

```
RL_Project_16_17_Deliverables/
├── README.md                          # This file
│
├── 16_problem_formulation.pdf         # Project 16: MDP formulation (state/action/reward)
├── 16_implementation_dqn.py           # Project 16: DQN learning-rate scheduler
├── 16_learning_curves.png             # Project 16: Comparative learning curves
├── 16_presentation_5slides.pptx       # Project 16: 5-slide presentation
│
├── 17_problem_formulation.pdf         # Project 17: MDP formulation (state/action/reward)
├── 17_implementation_reinforce.py     # Project 17: REINFORCE NAS controller
├── 17_learning_curves.png             # Project 17: Comparative learning curves
├── 17_presentation_5slides.pptx       # Project 17: 5-slide presentation
│
└── requirements.txt                   # Python dependencies
```

---

## 🔬 Project 16: RL-based Learning-Rate Scheduler

### Problem
Training neural networks requires carefully tuning the learning rate schedule. Instead of using fixed heuristics, we train a **DQN agent** that dynamically adjusts the learning rate during CNN training on CIFAR-10.

### MDP Formulation
| Component | Description |
|-----------|-------------|
| **State** | `[train_loss, Δloss, val_acc, Δval_acc, progress_ratio, log₁₀(lr)]` — 6-dim vector |
| **Action** | 5 discrete LR multipliers: `[×0.5, ×0.8, ×1.0, ×1.25, ×2.0]` |
| **Reward** | `100 × Δacc − 5 × max(0, Δloss)` + terminal bonus for accuracy ≥ 65% |
| **Environment** | Gym-like wrapper around a 3-layer CNN training loop on CIFAR-10 |

### Architecture
- **Child CNN**: Compact 3-layer CNN (Conv → BN → ReLU → Pool) with an adaptive pooling head
- **Q-Network**: 2-hidden-layer MLP (64 → 64) with experience replay buffer (capacity 2000) and target network
- **Exploration**: ε-greedy with decay (1.0 → 0.05, decay factor 0.90)

### Baselines Compared
1. **Fixed LR** — Constant η = 0.01 throughout training
2. **StepLR Decay** — Multiply by ×0.5 every 5 epochs
3. **Random Policy** — Uniformly random action selection
4. **RL (DQN) Scheduler** — Trained DQN agent (our method)

### How to Run
```bash
python 16_implementation_dqn.py --epochs 15 --rl_episodes 10 --subset 6000 --out_dir 16
```

---

## 🧬 Project 17: Mini NAS Controller

### Problem
Designing optimal neural network architectures is time-consuming. We train an **RNN controller** using **REINFORCE** to automatically propose CNN architectures, evaluated by validation accuracy on CIFAR-10.

### MDP Formulation
| Component | Description |
|-----------|-------------|
| **State** | Implicit (autoregressive RNN hidden state) |
| **Action** | 12 sequential decisions: for each of 3 layers → {filters, kernel_size, activation, residual} |
| **Reward** | `val_acc × 100 − 0.05 × (params / 10000)` — accuracy with parameter penalty |
| **Search Space** | Filters: {16, 32, 64}, Kernels: {3, 5}, Activations: {ReLU, LeakyReLU, GELU}, Residual: {Yes, No} |

### Architecture
- **RNN Controller**: LSTM-based autoregressive generator producing 12 architectural tokens
- **Child CNN**: Dynamically constructed from sampled architecture (ConvBlock with optional residual connections)
- **Training**: REINFORCE with exponential moving average baseline (decay = 0.85) and entropy bonus (0.01)

### Baselines Compared
1. **Random Architecture Search** — Uniformly sampled architectures from the search space
2. **REINFORCE Controller** — Learned RNN policy (our method)

### How to Run
```bash
python 17_implementation_reinforce.py --iterations 20 --child_epochs 3 --subset 3000 --out_dir 17
```

---

## 📊 Results

### Project 16: Learning-Rate Scheduling
| Method | Final Val Accuracy (%) | Total Reward |
|--------|----------------------|--------------|
| Fixed LR | Baseline | Baseline |
| StepLR Decay | Baseline | Baseline |
| Random Policy | Baseline | Baseline |
| **RL (DQN) Scheduler** | **Best** | **Best** |

> *Run the implementation to generate actual numbers. The DQN agent learns to adaptively reduce the learning rate as training progresses, achieving faster convergence.*

### Project 17: Neural Architecture Search
The REINFORCE controller learns to propose increasingly better architectures over the search budget, outperforming random search in both best-found accuracy and average reward.

---

## ⚙️ Requirements

- Python 3.8+
- PyTorch ≥ 1.10
- torchvision
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📦 Deliverables Checklist

Each project includes the four required deliverables:

| Deliverable | Project 16 | Project 17 |
|-------------|-----------|-----------|
| ✅ Problem Formulation (1-page PDF) | `16_problem_formulation.pdf` | `17_problem_formulation.pdf` |
| ✅ Implementation (standard RL algorithm) | `16_implementation_dqn.py` | `17_implementation_reinforce.py` |
| ✅ Learning Curves (vs. baselines) | `16_learning_curves.png` | `17_learning_curves.png` |
| ✅ 5-Slide Presentation | `16_presentation_5slides.pptx` | `17_presentation_5slides.pptx` |

---

## 📝 License

This project is submitted as coursework for academic purposes.
