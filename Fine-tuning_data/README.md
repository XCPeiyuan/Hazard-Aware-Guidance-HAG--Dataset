**LLaMA Factory fine-tuning files**
使用LLaMA-Factory

data文件夹中存放着数据集索引文件。将数据集与图片放入即可。

微调模型
```bash
llamafactory-cli train train/qwen2_5vl_lora_sft.yaml
```
合并模型
```bash
llamafactory-cli export merge/qwen2_5vl_lora_sft.yaml
```
