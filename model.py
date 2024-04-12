from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from auto_compressor import LlocoAutoCompressorModel
from peft import LoraConfig, get_peft_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IGNORE_INDEX = -100

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf")
    lora_r: int = field(
        default=8,
        metadata={"help": "lora rank"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "lora dropout"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "lora alpha"}
    )
    emb_model_name: str = field(
        default="",
        metadata={"help": "embedding model name"}
    )
    use_ft_token: bool = field(
        default=False,
        metadata={"help": "Whether to use a special token for sft."},
    )


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    instruction_data_path: str = field(default=None, metadata={"help": "Path to the instruction data."})
    embedding_path: str = field(default=None, metadata={"help": "Path to the preprocessed context embeddings."})
    debug_data: bool = field(default=False, metadata={"help": "Enable debug dataset to quickly verify the training process"})
    lazy_preprocess: bool = field(default=False, metadata={"help": "Whether to lazily preprocess the dataset."})
    eval_mode: str = field(default="baseline")
    dataset_name: str = field(default="qmsum")
    out_path: str = field(default=".")
    n_sample: int = field(default=10000, metadata={"help": "Number of samples to select from the training set."})
    max_new_tokens: int = field(default=50, metadata={"help": "Maximum number of new tokens to generate."})
    split: str = field(default="train", metadata={"help": "Dataset split to use."})
    needle_ctx_len: int = field(default=32000, metadata={"help": "Maximum context length."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )
    sources: str = field(
        default="all",
        metadata={"help": "Articles sources to train on."},
    )
    peft_model: str = field(
        default="",
    )
    exp_name: str = field(
        default="",
        metadata={"help": "Experiment name."},
    )


def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    print(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}")


def init_model(model_args, data_args, training_args):
    global local_rank
    local_rank = training_args.local_rank

    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = LlocoAutoCompressorModel.from_pretrained("princeton-nlp/AutoCompressor-Llama-2-7b-6k", 
                                                     torch_dtype=torch.bfloat16 if training_args.bf16 is True else torch.float16)
    model = get_peft_model(model, lora_config)
    model = model.to(device)
    rank0_print(model)

    return model