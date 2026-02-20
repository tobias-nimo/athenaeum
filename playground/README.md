# Examples

Interactive notebooks demonstrating how to use **Athenaeum**.

## Prerequisites

1. **Python 3.11+**
2. **An OpenAI API key** — set the `OPENAI_API_KEY` environment variable before running the notebook:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

## Quick start

```bash
pip install athenaeum-kb langchain-openai jupyter
jupyter notebook playground.ipynb
```

## Knowledge Base

The notebook uses PDFs from `knowledge-base/` — a curated set of landmark machine-learning papers:

| Topic                     | Archive Link                                                |
| ------------------------- | ----------------------------------------------------------- |
| **Attention-Transformer** | arXiv:1706.03762                                            |
| **LoRA**                  | arXiv:2106.09685                                            |
| **PEFT Surveys**          | arXiv:2312.12148, 2402.02242                                |
| **ViT**                   | arXiv:2010.11929                                            |
| **VAE**                   | arXiv:1312.6114                                             |
| **GAN**                   | arXiv:1406.2661                                             |
| **BERT**                  | arXiv:1810.04805                                            |
| **Diffusion Models**      | arXiv:2006.11239, 2112.10752                                |
| **RAG**                   | arXiv:2005.11401                                            |
| **GPT-1/2/3**             | OpenAI/GPT-1 deriv link; arXiv:1906.08237; arXiv:2005.14165 |

