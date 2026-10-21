# ShoppingComp Data Notice

This directory contains a RAG-oriented conversion of the Chinese subset of
ShoppingComp.

## Source

- Dataset: ShoppingComp
- Publisher: huaixiao
- Data card: <https://hf-mirror.com/datasets/huaixiao/ShoppingComp>
- Paper: <https://www.arxiv.org/abs/2511.22978>
- License: [MIT](https://opensource.org/license/mit)

ShoppingComp contains 97 ground-truth shopping tasks and 48 safety-critical
tasks, released in both English and Chinese. This directory uses only the
Chinese files. The original data card identifies 42 tasks as expert questions
and 55 tasks as synthesized user questions. Product evidence includes source
URLs, but those external sources should be checked before production use.

## Conversion

The converter produces 1190 product-evidence documents from 236 products and
keeps 97 shopping questions separate from the knowledge base. It removes URLs
from the BM25/embedding text, canonicalizes and deduplicates source links, and
stores source metadata separately. It also adds stable RAG identifiers,
expected product/document/chunk references, and 48 separate safety tasks.
Scene rubrics are retained in the raw input rather than added to retrieval
documents to reduce answer leakage.

Run the converter from the project root:

```powershell
uv run python scripts\\prepare_shoppingcomp.py
```

The converted files remain available under the MIT License.

## Citation

```bibtex
@article{shoppingcomp2025,
  title={ShoppingComp: Are LLMs Really Ready for Your Shopping Cart?},
  journal={arXiv preprint arXiv:2511.22978},
  year={2025}
}
```
