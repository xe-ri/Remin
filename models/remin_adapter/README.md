# Remin LoRA Adapter Directory

This directory is the default local output path for the Remin LoRA adapter.

Model weights and tokenizer files are intentionally not committed to GitHub because they are large generated artifacts. After training, this directory may contain files such as:

- `adapter_config.json`
- `adapter_model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`
- `checkpoint-*`

To regenerate the adapter, run from the project root:

```bash
python training/train_lora.py --base-model unsloth/Qwen3-8B-bnb-4bit
```

When running the backend, keep `MODEL_PATH=models/remin_adapter` in `.env` if you want the RAG service to load the adapter from this directory. If the base model is stored locally, also set `LOCAL_BASE_MODEL_PATH` or `BASE_MODEL_PATH`.
