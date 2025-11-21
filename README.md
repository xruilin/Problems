#Daily Problems

# ML Research Engineering Handbook & Environment Setup

This repository serves as a collection of best practices, environment configuration scripts, and troubleshooting notes for Deep Learning and Reinforcement Learning research. 

It is designed to streamline the process of setting up complex environments involving **PyTorch**, **CUDA**, **vLLM**, and custom kernels like **FlashAttention**.

## 📚 Table of Contents

- [Conda & Environment Management](#conda--environment-management)
- [GPU & CUDA Configuration](#gpu--cuda-configuration)
- [Advanced Installation Guides](#advanced-installation-guides)
  - [FlashAttention (Source Compilation)](#flashattention-source-compilation)
  - [vLLM Deployment](#vllm-deployment)
- [Docker Best Practices](#docker-best-practices)
- [Productivity Tools](#productivity-tools)

---

## Conda & Environment Management

### Avoid Path Hardcoding
When moving environments between storage mounts (e.g., `/mnt/usercache` to `/netcache`), hardcoded shebangs in `conda` executables often break.

**Recommended Activation Method:**
Instead of `source ./bin/activate`, use the initializing script to properly load paths:
```bash
source /path/to/anaconda3/etc/profile.d/conda.sh
conda activate <env_name>
