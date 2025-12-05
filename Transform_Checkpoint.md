这份笔记整理了该脚本的功能逻辑、核心代码以及关键实现细节，主要用于解决**分布式训练（如 FSDP/VeRL）生成的 Checkpoint 无法直接被 Hugging Face Transformers 加载**的问题。

-----

# 🛠️ 脚本笔记：PyTorch 分布式权重 (DTensor) 转 Hugging Face 格式

## 1\. 脚本背景与目标

在使用 PyTorch 分布式训练框架（如 FSDP 或基于 VeRL 的强化学习训练）时，保存的权重往往是 **分片（Sharded）** 的，并且数据类型可能是 `DTensor` (Distributed Tensor)。

  - **问题**：标准的 `AutoModel.from_pretrained` 无法识别 `DTensor` 对象，也无法直接读取分散在多个文件中的 `rank_*.pt` 权重。
  - **目标**：将分散的、包含 `DTensor` 的权重文件清洗、解包、合并，并转换为标准的 Hugging Face 格式，以便进行推理或继续微调。

-----

## 2\. 完整代码

```python
import os
import torch
import glob
import json
from transformers import AutoModelForCausalLM, AutoConfig

# --- 尝试导入分布式相关库 ---
try:
    from torch.distributed._tensor import DTensor
except ImportError:
    DTensor = None
    print("Warning: torch.distributed._tensor.DTensor not found. Type checking might rely on string matching.")

# --- 配置 ---
INPUT_DIR = "/mnt/usercache/xuruilin/verl-agent/checkpoints/verl_agent_grammar/grpo_qwen3_4b_grammar/global_step_186/actor"
OUTPUT_DIR = "/mnt/usercache/xuruilin/verl-agent/checkpoints/verl_agent_grammar/grpo_qwen3_4b_grammar/final/actor_fixed"

def unwrap_dtensor(t):
    """
    强制解包 DTensor。
    如果 t 是 DTensor，调用 to_local() 获取本地分片。
    """
    # 1. 显式类型检查
    if DTensor is not None and isinstance(t, DTensor):
        return t.to_local()
    
    # 2. 字符串类型检查 (备用，防止 import 失败或类名匹配问题)
    if 'DTensor' in type(t).__name__:
        if hasattr(t, 'to_local'):
            return t.to_local()
        else:
            print(f"Warning: Found DTensor-like object {type(t)} but no to_local() method.")
            
    return t

def main():
    print(f"Converting {INPUT_DIR} -> {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载 Config
    print("Loading config...")
    try:
        config = AutoConfig.from_pretrained(INPUT_DIR)
    except:
        with open(os.path.join(INPUT_DIR, "config.json"), "r") as f:
            config = AutoConfig.from_dict(json.load(f))

    # 2. 读取所有分片
    shards = sorted(glob.glob(os.path.join(INPUT_DIR, "model_world_size_*_rank_*.pt")))
    if not shards:
        raise FileNotFoundError(f"No checkpoint shards found in {INPUT_DIR}")
    print(f"Found {len(shards)} shards.")

    full_state_dict = {}

    # 3. 逐个加载分片并清洗
    for i, shard in enumerate(shards):
        print(f"Processing shard {i+1}/{len(shards)}: {os.path.basename(shard)}...")
        # 强制加载到 CPU，避免显存爆炸
        shard_data = torch.load(shard, map_location="cpu", weights_only=False)
        
        clean_shard = {}
        for k, v in shard_data.items():
            # [核心修复] 解包 DTensor
            clean_v = unwrap_dtensor(v)
            clean_shard[k] = clean_v

        # --- 简单的合并逻辑 ---
        # 假设分片主要是在 dim=0 上进行的 (FSDP 常见切分方式之一)
        for k, v in clean_shard.items():
            if k in full_state_dict:
                # 尝试拼接
                try:
                    if isinstance(v, torch.Tensor) and isinstance(full_state_dict[k], torch.Tensor):
                        if full_state_dict[k].dim() > 0:
                            full_state_dict[k] = torch.cat([full_state_dict[k], v], dim=0)
                        else:
                            pass 
                except Exception as e:
                    pass
            else:
                full_state_dict[k] = v

    print(f"Merged state dict has {len(full_state_dict)} keys.")
    
    # 4. 移除 DTensor 残留并修正 Key (去除 module. 前缀)
    final_state_dict = {}
    for k, v in full_state_dict.items():
        if 'DTensor' in type(v).__name__:
             v = unwrap_dtensor(v)
             
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        final_state_dict[new_k] = v

    # 5. 加载到 HF 模型
    print("Loading into AutoModelForCausalLM...")
    # 这里的 with torch.device("meta") 是为了加速初始化，但随后还是重新实例化了
    # 实际使用的是下面的 AutoModelForCausalLM.from_config(config)
    model = AutoModelForCausalLM.from_config(config)
    
    try:
        model.load_state_dict(final_state_dict, strict=False)
    except RuntimeError as e:
        print("\n!!!!!!!! LOAD ERROR !!!!!!!!")
        print(str(e)[:500])
    
    # 6. 保存
    print(f"Saving to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    config.save_pretrained(OUTPUT_DIR)
    
    print("Copying tokenizer files...")
    # 复制 tokenizer 相关文件以确保模型可直接使用
    tokenizer_files = glob.glob(os.path.join(INPUT_DIR, "tokenizer*")) + \
                      glob.glob(os.path.join(INPUT_DIR, "vocab*")) + \
                      glob.glob(os.path.join(INPUT_DIR, "merges*")) + \
                      glob.glob(os.path.join(INPUT_DIR, "added_tokens*")) + \
                      glob.glob(os.path.join(INPUT_DIR, "chat_template*")) + \
                      glob.glob(os.path.join(INPUT_DIR, "special_tokens_map*"))
    for f in tokenizer_files:
        os.system(f"cp '{f}' '{OUTPUT_DIR}'")

    print("Done!")

if __name__ == "__main__":
    main()
```

-----

## 3\. 功能模块详细解析

### A. 核心难点解决：`DTensor` 解包

代码中最关键的函数是 `unwrap_dtensor(t)`。

  * **背景**：PyTorch 分布式框架会把模型权重包装成 `DTensor`，其中包含了 global shape 和 local shard 的元数据。如果直接保存，加载时会报错。
  * **实现**：
    1.  **显式检查**：首先尝试导入 `DTensor` 类进行 `isinstance` 检查。
    2.  **隐式检查（容错）**：如果环境缺包或类型匹配失败，检查类型名称字符串是否包含 "DTensor"。
    3.  **动作**：调用 `.to_local()` 方法，丢弃分布式元数据，将其转化为普通的 `torch.Tensor`。

### B. 权重文件搜寻与加载

  * **路径模式**：脚本寻找匹配 `model_world_size_*_rank_*.pt` 的文件。这是典型的分布式训练保存格式（World Size 代表总进程数，Rank 代表当前进程 ID）。
  * **CPU 加载**：`map_location="cpu"` 非常重要，因为合并后的模型可能非常大，且清洗过程不需要 GPU 计算，这样可以避免显存溢出 (OOM)。

### C. 朴素合并策略 (Merge Strategy)

脚本采用了一种简化的合并逻辑：

  * **逻辑**：遍历所有 shard 文件，如果发现同一个 key (参数名) 在之前的 shard 中已存在，则使用 `torch.cat(..., dim=0)` 进行拼接。
  * **适用性**：这是一种启发式（Heuristic）方法。它假设分布式切分是沿着第 0 维进行的（例如 Embedding 层或某些 Linear 层的行切分）。
  * **注意**：对于复杂的张量并行（Tensor Parallelism），某些层可能是列切分（dim=1），这种简单的 `dim=0` 拼接可能会导致权重错位。但在脚本的特定上下文（可能是 FSDP ZeRO-3 某些设置）下，这种方法被作为首选方案。

### D. 键名清洗 (State Dict Cleaning)

  * **去前缀**：分布式训练通常会在模型外层包裹 `DistributedDataParallel` 或类似的 Wrapper，导致参数名带有 `module.` 前缀。
  * **修正**：`k.replace("module.", "")` 确保参数名能和 Hugging Face 的 `config` 对应上。

### E. 最终转换与保存

1.  **初始化空模型**：利用 `AutoConfig` 和 `AutoModelForCausalLM.from_config` 创建一个骨架模型（随机初始化权重）。
2.  **加载权重**：使用 `load_state_dict(..., strict=False)` 将合并清洗后的权重填入骨架。`strict=False` 允许部分非关键参数缺失（例如缓冲区 buffer）。
3.  **保存**：调用 HF 标准的 `save_pretrained`，将模型保存为 `model.safetensors` (或 bin) 和 `config.json`。
4.  **迁移 Tokenizer**：脚本最后通过 shell 命令 (`cp`) 暴力复制所有 tokenizer 相关文件，确保输出目录是一个完整的、可直接加载的模型包。

-----

## 4\. 使用注意事项

1.  **环境依赖**：最好在包含 `torch` 和 `transformers` 的环境中运行。如果原始环境使用了特殊的分布式库（如 `verl`），最好在该环境中运行以确保 `DTensor` 能正确 import。
2.  **内存消耗**：脚本在 CPU 内存中构建了完整的 `full_state_dict`。对于超大模型（如 70B+），这可能导致内存溢出。
3.  **合并风险**：如前所述，`dim=0` 的合并是假设。如果转换后的模型输出乱码或 perplexity 极高，通常是由于合并维度错误（应为 dim=1）导致的。
