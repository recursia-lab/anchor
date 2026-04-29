# Google 触达策略草稿

## 渠道 1：PaliGemma2 官方仓库 Discussion

**目标仓库：** https://github.com/google-research/big_vision 或 https://github.com/google/paligemma

**标题：**
```
Community ecosystem update: Anchor (multi-LoRA serving) + SGLang PR + Unsloth PR
```

**内容：**
```
Hi PaliGemma team,

We've been building out the PaliGemma2 open-source ecosystem and wanted to share what we've shipped:

1. **Anchor** — multi-LoRA serving framework, all adapters in VRAM, 216ms switching
   https://github.com/recursia-lab/anchor

2. **SGLang PR #24034** — native PaliGemma2 + LoRA registration
   https://github.com/sgl-project/sglang/pull/24034

3. **Unsloth PR #5218** — PaliGemma v1+v2 LoRA fine-tuning (23 lines, class-level patching)
   https://github.com/unslothai/unsloth/pull/5218

4. **Python client** — `pip install anchor-vision`

5. **Community adapter hub** — https://github.com/recursia-lab/paligemma2-adapters

We're a small team building industrial vision inspection systems on top of PaliGemma2.
Happy to contribute more — open to suggestions on what the community needs most.
```

---

## 渠道 2：Google for Startups Cloud Program

**申请链接：** https://cloud.google.com/startup

**申请理由草稿：**
```
Company: Recursia Lab
Stage: Early (pre-revenue, building MVP)
Use case: Industrial vision inspection using PaliGemma2 + LoRA fine-tuning
Cloud usage: Cloud Run GPU (L4), Vertex AI training

Open-source contributions:
- Anchor: PaliGemma2 multi-LoRA serving (GitHub: recursia-lab/anchor)
- SGLang PR #24034: native PaliGemma2 support
- Unsloth PR #5218: PaliGemma LoRA fine-tuning

We are building production AI infrastructure on top of Google's PaliGemma2 and
contributing back to the open-source ecosystem. Cloud credits would help us:
1. Keep our serving infrastructure running (Cloud Run GPU ~$50/month)
2. Train more LoRA adapters on Vertex AI
3. Expand the open-source tooling
```

---

## 渠道 3：HuggingFace Blog Post

**标题：**
```
Building PaliGemma2 ecosystem: From 0 to multi-LoRA serving in production
```

**结构：**
1. 为什么选 PaliGemma2（工业视觉检测场景）
2. 我们发现的生态缺口
3. 我们做了什么（Anchor / SGLang / Unsloth）
4. 技术亮点（216ms switching，23行代码解锁 Unsloth 支持）
5. 下一步（llama.cpp，Ollama）
6. 欢迎社区贡献

**发布平台：**
- HuggingFace Blog (需要 HF 账号 write 权限)
- Medium (可以直接发)
- dev.to

---

## 渠道 4：Twitter/X 发布节奏

每个 PR merge 时发一条：

**SGLang merge 时：**
```
We just got PaliGemma2 native LoRA support merged into @lmsysorg SGLang 🎉

PR #24034 adds:
- PaliGemmaForConditionalGeneration model registration  
- packed qkv/gate_up mapping
- lora_pattern for language_model layers

Full ecosystem: github.com/recursia-lab/anchor
@GoogleDeepMind
```

**Unsloth merge 时：**
```
23 lines to unlock PaliGemma2 LoRA fine-tuning in @UnslothAI 🔓

The trick: FastGemma2Model.pre_patch() patches at class level →
PaliGemma2's language_model auto-inherits all fast kernels.

PR #5218: github.com/unslothai/unsloth/pull/5218
@GoogleDeepMind
```
