# Qwen3-VL Environment Setup Guide

This document outlines the specific environment configuration and installation steps required to run **Qwen3-VL** with **vLLM 0.11.0**.

It addresses compatibility with **PyTorch 2.8.0** and provides the specific compilation flags needed for **FlashAttention**.

## 📋 Version Requirements

| Package | Version | Note |
| :--- | :--- | :--- |
| **Python** | `3.12` | Recommended |
| **PyTorch** | `2.8.0` | Core dependency |
| **vLLM** | `0.11.0` | Inference engine |
| **Transformers** | `4.57.1` | HF ecosystem |
| **CUDA** | `12.x` | Ensure compatibility with Torch 2.8 |

---

## 🚀 Installation Steps

### 1. Create and Activate Environment

Start with a fresh Conda environment to avoid conflicts.

```bash
conda create -n qwen_vl python=3.12 -y
conda activate qwen_vl

# Install PyTorch 2.8.0
# Note: Ensure you have the correct CUDA version (e.g., cu121, cu124, or cu128)
pip install torch==2.8.0

# Install vLLM and Transformers
pip install vllm==0.11.0 transformers==4.57.1

# 1. Clean cache to prevent using corrupted downloads
pip cache purge

# 2. Compile and Install
MAX_JOBS=16 FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn --no-build-isolation
