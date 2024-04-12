import copy
import os
from typing import Dict, Optional

import torch
import transformers
from data import make_lloco_data_module
from datasets import load_dataset
from model import DataArguments, ModelArguments, TrainingArguments, init_model
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IGNORE_INDEX = -100
truncation_seperator = "... [The rest of the story is omitted]\n\n"
local_rank = None


hqa_prompt = "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


class LazyHotpotSFTDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        embedding_path: Optional[str] = None,
        inference_mode: bool = False,
        split: str = "train",
        mode: str = "baseline",
        n_sample: int = 10000,
    ):
        super(LazyHotpotSFTDataset, self).__init__()
        rank0_print("Loading data...")
        if split == "validation":
            self.dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")
            self.dataset = self.__preproc_dataset(self.dataset)
            print("dataset size:", len(self.dataset))
        else:
            self.dataset = load_dataset("hotpot_qa", "fullwiki", split="train")
            rank0_print(f"Shuffling and selecting {n_sample} examples from the training set...")
            self.dataset = self.dataset.shuffle(seed=42).select(range(n_sample))

        if embedding_path is not None:
            rank0_print("Loading context embeddings...")
            self.context_embeddings_map = torch.load(embedding_path)
            self.is_preprocessed = True
        else:
            rank0_print(
                "No context embeddings provided, will use context data instead."
            )
            self.context_embeddings_map = None
            self.is_preprocessed = False

        self.tokenizer = tokenizer
        self.cached_data_dict = {}
        self.inference_mode = inference_mode
        self.mode = mode

        print("Current mode:", self.mode)

    def __preproc_dataset(self, dataset):
        ret = []
        visited = set()
        for data in tqdm(dataset):
            example_id = data["id"]
            if example_id not in visited:
                visited.add(example_id)
                ret.append(data)
        return ret

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        entry = self.dataset[i]
        article_id = entry["id"]

        question = entry["question"]
        answer = entry["answer"]

        icl_prompt = "You were just given an article from above. I will now give you a question. Answer the question as concisely as you can, using a single phrase or sentence if possible.\nQuestion: "

        question = icl_prompt + question + "\nAnswer:"

        q_input_ids = self.tokenizer(
            question, truncation=True, add_special_tokens=False
        ).input_ids
        a_input_ids = self.tokenizer(
            answer, truncation=True, add_special_tokens=False
        ).input_ids
        a_input_ids += [self.tokenizer.eos_token_id]

        if "baseline" in self.mode:
            if self.mode == "baseline_nocontext":
                start_index = hqa_prompt.find("{context}\n\n") + len("{context}\n\n")
                prompt = hqa_prompt[start_index:].format(
                    input=entry["question"],
                )
            elif self.mode == "baseline":
                context = ""
                for i, (title, sentences) in enumerate(
                    zip(entry["context"]["title"], entry["context"]["sentences"])
                ):
                    if i > 0:
                        context += "\n\n"
                    context += title + "\n"
                    for sent in sentences:
                        context += sent

                prompt = hqa_prompt.format(
                    input=question,
                    context=context,
                )
            prompt = f"[INST]{prompt}[/INST]"
            decoder_input_ids = self.tokenizer(
                prompt,
                padding="longest",
                truncation=True,
                add_special_tokens=False,
                max_length=4000,
            ).input_ids
        else:
            decoder_input_ids = copy.deepcopy(q_input_ids)

        if not self.inference_mode:
            decoder_input_ids += a_input_ids

        decoder_input_ids = torch.as_tensor(decoder_input_ids)
        labels = copy.deepcopy(decoder_input_ids)
        labels[: len(q_input_ids)] = IGNORE_INDEX

        assert self.is_preprocessed
        context_embeddings = self.context_embeddings_map[article_id]

        if self.inference_mode:
            ret = dict(
                decoder_input_ids=decoder_input_ids.to(device),
                context_embeddings=context_embeddings.to(device),
            )
            return ret
        else:
            ret = dict(
                input_ids=decoder_input_ids,
                labels=labels,
                inputs_embeds=context_embeddings,
            )
        return ret

    def get_ground_truth(self, i):
        return self.dataset[i]["answer"]

    def get_example_id(self, i):
        return self.dataset[i]["id"]


def train():
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    if not os.path.exists(data_args.embedding_path):
        rank0_print("Embedding file does not exist...")
        exit()
    else:
        rank0_print("Embedding file exists, skipping preprocessing...")

    model = init_model(model_args, data_args, training_args)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = '[PAD]'

    model.config.use_cache = False  # required for gradient checkpointing
    model.base_model.enable_input_require_grads()  # required for gradient checkpointing
    model.base_model.gradient_checkpointing_enable()  # enable gradient

    data_module = make_lloco_data_module(model=model,
                                        tokenizer=tokenizer,
                                        data_args=data_args,
                                        dataset_cls=LazyHotpotSFTDataset,
                                        n_sample=data_args.n_sample,
                                        )
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
