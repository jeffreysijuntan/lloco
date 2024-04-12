import json
import os
import sys

import torch
import transformers
from peft import PeftModel
from finetune_hotpot import LazyHotpotSFTDataset
from finetune_quality import LazyQualitySFTDataset
from finetune_scrolls import LazyScrollsSFTDataset
from auto_compressor import LlocoAutoCompressorModel
from model import DataArguments, ModelArguments, TrainingArguments
from tqdm import tqdm
from vllm import LLM, SamplingParams

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.inference_mode()
def eval_scrolls():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    max_tokens = data_args.max_new_tokens

    eval_mode = data_args.eval_mode
    if eval_mode == "baseline" or eval_mode == "baseline_nocontext":
        sampling_params = SamplingParams(max_tokens=max_tokens)
        model = LLM(model=model_args.model_name_or_path)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path
        )
        tokenizer.pad_token = "[PAD]"
    else:
        model = LlocoAutoCompressorModel.from_pretrained(
            "princeton-nlp/AutoCompressor-Llama-2-7b-6k", 
            torch_dtype=torch.bfloat16 if training_args.bf16 is True else torch.float16)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        tokenizer.pad_token = '[PAD]'
        print("Loading Peft model...")
        model = PeftModel.from_pretrained(
            model.to(device),
            training_args.peft_model,
            torch_dtype=torch.float16,
        )

    print("=" * 50)
    print(f"{model} model loaded!, device:{device}")
    
    if data_args.dataset_name in ["narrative_qa", "qasper", "qmsum"]:
        dataset = LazyScrollsSFTDataset(
            tokenizer=tokenizer,
            embedding_path=data_args.embedding_path,
            dataset_name=data_args.dataset_name,
            split="validation",
            mode=data_args.eval_mode,
        )
    elif data_args.dataset_name == "quality":
        dataset = LazyQualitySFTDataset(
            tokenizer=tokenizer,
            quality_path=data_args.data_path,
            embedding_path=data_args.embedding_path,
            split="validation",
            mode=data_args.eval_mode,
        )
    elif data_args.dataset_name == "hotpot_qa":
        dataset = LazyHotpotSFTDataset(
            tokenizer=tokenizer,
            embedding_path=data_args.embedding_path,
            split="validation",
            inference_mode=True,
            mode=data_args.eval_mode
        )

    total = 0
    res = {}
    for i, entry in enumerate(tqdm(dataset)):
        total += 1

        if eval_mode == "baseline" or eval_mode == "baseline_nocontext":
            output_ids = model.generate(
                prompt_token_ids=[entry["decoder_input_ids"]],
                sampling_params=sampling_params,
            )[0]
            predicted_text = output_ids.outputs[0].text.strip()
        else:
            prompt_len = entry["decoder_input_ids"].size(0)
            output_ids = model.generate(input_ids=entry["decoder_input_ids"].unsqueeze(0), 
                                        softprompt=entry["context_embeddings"].unsqueeze(0).to(torch.float16), 
                                        max_new_tokens=max_tokens)[0]
            predicted_text = tokenizer.decode(
                output_ids[prompt_len:], skip_special_tokens=True
            )

        ground_truth = dataset.get_ground_truth(i)

        print("----------------Predicted text ----------------\n", predicted_text)
        print("---------------- Ground truth -----------------\n", ground_truth)
        print("\n\n")

        example_id = dataset.get_example_id(i)
        res[example_id] = predicted_text

    with open(data_args.out_path, "w+") as f:
        json.dump(res, f)


if __name__ == "__main__":
    eval_scrolls()
