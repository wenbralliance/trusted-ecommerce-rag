---
license: mit
task_categories:
- question-answering
language:
- en
- zh
tags:
- agent
- e-commerce
pretty_name: shoppingcomp
size_categories:
- n<1K
---

## Dataset Summary

**ShoppingComp** is a benchmark dataset for evaluating LLM-powered online shopping agents under realistic, consumer-driven, and safety-critical settings.  
It is introduced in [`ShoppingComp: Are LLMs Really Ready for Your Shopping Cart?`](https://www.arxiv.org/abs/2511.22978) 

The dataset targets **open-world shopping**, where agents must retrieve products from the web, verify multi-constraint requirements, reason over noisy or conflicting evidence, and recognize unsafe or inappropriate product usage.

ShoppingComp is designed **for evaluation rather than training**.

---

## Dataset Composition

ShoppingComp contains **145 tasks per language**, with two complementary subsets:

- **Ground-Truth Tasks (97)**: Standard shopping tasks with expert-annotated scenarios, fine-grained evaluation rubrics, reference product lists, and evidence-based rationales.
- **Safety-Critical Tasks (48)**: Tasks containing explicit safety or misuse traps, designed to evaluate risk-aware decision making and safety compliance.

All tasks are released in **both English and Chinese**, resulting in **290 total task instances** across two languages.

---

## Files

This release contains 4 JSONL files (recommended to keep filenames unchanged when publishing on Hugging Face):

- `ShoppingComp_97_20260127.en.jsonl`: English GT (97)
- `ShoppingComp_97_20260127.zh.jsonl`: Chinese GT (97)
- `ShoppingComp_traps_48.en.jsonl`: English traps/safety (48)
- `ShoppingComp_traps_48.zh.jsonl`: Chinese traps/safety (48)

Per `ShoppingCompJudge/README.md`, the `data/` folder in this repo is the **latest version** (97 + 48 examples). The paper reported results on 120 examples; paper metrics will be updated. **Please treat this repo as the source of truth.**

---

## Data Format

The dataset is released in **JSONL** format.

### Ground-Truth Tasks

Each ground-truth example includes:
- `uuid`: unique task identifier  
- `question`: user shopping request  
- `scene_list`: multiple shopping scenarios with scenario-level rubrics  
- `product_list`: expert-annotated reference products, with per-scenario rationale and evidence  

### Safety-Critical Tasks

Each safety task includes:
- `uuid`: unique task identifier  
- `question`: user shopping request  
- `trap_rubric`: safety or misuse constraint used for evaluation  

---

## Usage

ShoppingComp supports evaluation of shopping agents along multiple dimensions, including retrieval coverage over correct products, satisfaction of fine-grained constraints, reasoning consistency and evidence grounding, and safety awareness.

Load with 🤗 `datasets`:

```python
from datasets import load_dataset

data_files = {
    "gt_en": "ShoppingComp_97_20260127.en.jsonl",
    "gt_zh": "ShoppingComp_97_20260127.zh.jsonl",
    "traps_en": "ShoppingComp_traps_48.en.jsonl",
    "traps_zh": "ShoppingComp_traps_48.zh.jsonl",
}

ds = load_dataset("json", data_files=data_files)
```

---

## Intended Use and Limitations

- **Intended use**: benchmarking and analysis of LLM-based shopping agents in realistic, open-world, and safety-critical scenarios.
- **Not intended for**: model pretraining or large-scale fine-tuning.
- **Limitations**: limited dataset size; annotations reflect expert judgment and may not capture all regional, temporal, or market-specific product variations.

---

## Citation

If you use ShoppingComp, please cite the paper:

```bibtex
@article{shoppingcomp2025,
  title={ShoppingComp: Are LLMs Really Ready for Your Shopping Cart?},
  journal={arXiv preprint arXiv:2511.22978},
  year={2025}
}
```

---

## License

This dataset is released under the **MIT License**.