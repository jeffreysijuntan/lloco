import copy
import os
from typing import Dict, Optional

import torch
import transformers
from data import make_lloco_data_module
from model import DataArguments, ModelArguments, TrainingArguments, init_model
from torch.utils.data import Dataset
from transformers import Trainer
from utils import load_jsonl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IGNORE_INDEX = -100
truncation_seperator = "... [The rest of the story is omitted]\n\n"
local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


quality_prompt = "You are provided a story from above. We will now give you a multiple-choice question with 4 possible answers (marked by A, B, C, D). Choose the best answer by writing its corresponding letter (either A, B, C, or D).\nExample question:\nWhere is the capital of France?\n\nChoices:\nA. Berlin\nB. Paris\nC. London\nD. Tokyo\nChoose the best answer by writing its corresponding letter (either A, B, C, or D).\n\nAnswer:\nB. Paris\n\n"


class LazyQualitySFTDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        quality_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        embedding_path: Optional[str] = None,
        split: str = "train",
        mode: str = "baseline",
    ):
        super(LazyQualitySFTDataset, self).__init__()
        rank0_print("Loading data...")
        self.quality_data = load_jsonl(quality_path)
        self.context_data_dict = self.__build_context_dict(self.quality_data)
        self.sft_data_dict = self.__preprocess(self.quality_data)

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
        assert self.is_preprocessed

        self.tokenizer = tokenizer
        self.is_eval = True if split == "validation" else False
        self.eval_mode = mode

    def __build_context_dict(self, quality_data):
        ret = {}
        for entry in quality_data:
            article_id = entry["article_id"]
            article = entry["article"]
            ret[article_id] = article
        return ret

    def __preprocess(self, quality_data):
        ret = []
        choice_prefix = ["A. ", "B. ", "C. ", "D. "]
        for entry in quality_data:
            for q in entry["questions"]:
                res = {}

                gold_label = q["gold_label"]

                gt = gold_label - 1

                question_prompt = "Question:\n" + q["question"] + "\n\n"

                choice_prompt = "Choices:\n"
                choices = q["options"]
                for i in range(4):
                    c = choices[i]
                    choice_prompt += choice_prefix[i] + c + "\n"
                choice_prompt += (
                    "Choose the best answer by writing its corresponding letter (either A, B, C, or D).\n\nAnswer:"
                    + "\n"
                )
                answer_prompt = choice_prefix[gt] + choices[gt]
                res = {
                    "article_id": entry["article_id"],
                    "id": q["question_unique_id"],
                    "question": question_prompt + choice_prompt,
                    "answer": answer_prompt,
                }
                ret.append(res)
        return ret

    def __len__(self):
        return len(self.sft_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        instruction_pair = self.sft_data_dict[i]
        article_id = instruction_pair["article_id"]
        question = instruction_pair["question"]
        answer = instruction_pair["answer"]

        q_prompt = quality_prompt + question
        q_input_ids = self.tokenizer(
            q_prompt,
            add_special_tokens=False,
        ).input_ids
        a_input_ids = self.tokenizer(answer, add_special_tokens=False).input_ids
        a_input_ids += [self.tokenizer.eos_token_id]

        context_embeddings = self.context_embeddings_map[article_id]

        if self.is_eval:
            if self.eval_mode == "baseline":
                context = self.context_data_dict[article_id]
                context = "Article: " + context + truncation_seperator
                c_input_ids = self.tokenizer(
                    context,
                    padding="longest",
                    truncation=True,
                    add_special_tokens=False,
                    max_length=4000 - len(q_input_ids),
                ).input_ids
                decoder_input_ids = copy.deepcopy(c_input_ids + q_input_ids)
                return {
                    "decoder_input_ids": torch.as_tensor([decoder_input_ids]).to(device)
                }
            elif self.eval_mode == "baseline_nocontext":
                decoder_input_ids = copy.deepcopy(q_input_ids)
                return {
                    "decoder_input_ids": torch.as_tensor([decoder_input_ids]).to(device)
                }
            else:
                decoder_input_ids = copy.deepcopy(q_input_ids)
                ret = dict(
                    decoder_input_ids=torch.as_tensor(decoder_input_ids).to(device),
                    context_embeddings=context_embeddings.to(device),
                )
            return ret
        else:
            decoder_input_ids = copy.deepcopy(q_input_ids)
            decoder_input_ids += a_input_ids

            decoder_input_ids = torch.as_tensor(decoder_input_ids)
            labels = copy.deepcopy(decoder_input_ids)
            labels[: len(q_input_ids)] = IGNORE_INDEX

            ret = dict(
                input_ids=decoder_input_ids,
                labels=labels,
                inputs_embeds=context_embeddings,
            )
        return ret

    def get_ground_truth(self, i):
        instruction_pair = self.sft_data_dict[i]
        answer = instruction_pair["answer"]
        return answer
    
    def get_example_id(self, i):
        return self.sft_data_dict[i]["id"]


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
        dataset_cls=LazyQualitySFTDataset,
        data_args=data_args,
        quality_path=data_args.data_path,
    )
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
