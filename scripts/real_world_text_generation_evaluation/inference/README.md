# Historical Inference Scripts

These scripts produced the three-line prediction files used by the sibling
Real-world text-generation evaluation scripts. They are preserved as original
experiment snapshots and are not normalized into a single runner.

| Script | Historical setting |
| --- | --- |
| `inference_txt_based.py` | Local R-only Qwen2.5-VL inference. |
| `inference_txt_based_sp_token.py` | Local R+U Qwen2.5-VL 3B inference with special-token cleanup. |
| `inference_txt_based-prompt-engineering.py` | Qwen2.5-VL prompt-only baseline inference. |
| `inference_txt_based-prompt-engineering-gpt.py` | GPT-4o prompt-only baseline inference. |

The GPT script contains an API-key placeholder only. Supply a valid key outside
version control. Do not commit a real API key, model weights, images, or output
files.
