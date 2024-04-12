import copy
import os
from typing import Dict, Optional

import torch
import transformers
from datasets import load_dataset
from model import DataArguments, ModelArguments, TrainingArguments, init_model
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import Trainer
from data import make_lloco_data_module
from utils import load_jsonl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IGNORE_INDEX = -100
truncation_seperator = "... [The rest of the story is omitted]\n\n"
local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]

quality_prompt = "You are provided a story from above. We will now give you a multiple-choice question with 4 possible answers (marked by A, B, C, D). Choose the best answer by writing its corresponding letter (either A, B, C, or D).\nExample question:\nWhere is the capital of France?\n\nChoices:\nA. Berlin\nB. Paris\nC. London\nD. Tokyo\nChoose the best answer by writing its corresponding letter (either A, B, C, or D).\n\nAnswer:\nB. Paris\n\n"

qasper_prompt = "You are just given an scientific article from above. I will now give you a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write 'unanswerable'. If the question is a yes/no question, answer 'yes', 'no', or 'unanswerable'.\nQuestion: "

qmsum_prompt = "You are given a meeting transcript from above. I will now give you a query containing a question or instruction. Answer the query in one or more sentences. \nQuery: "

nqa_prompt = "You are given a story from above, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible.\nQuestion:"

gov_prompt = "You are given a report by a government agency. Write a one-page summary of the report."

musique_prompt = "You are given several paragraphs from Wikipedia and a question. Answer the question as concisely as you can, using a single phrase if possible. If the question cannot be answered based on the information in the paragraphs, write 'unanswerable'."

sys_prompts = {
    "quality": quality_prompt,
    "qasper": qasper_prompt,
    "qmsum": qmsum_prompt,
    "narrative_qa": nqa_prompt,
    "gov_report": gov_prompt,
}

scrolls_datasets = [
    "gov_report",
    "summ_screen_fd",
    "qmsum",
    "squality",
    "qasper",
    "narrative_qa",
    "quality",
    "musique",
    "space_digest",
    "book_sum_sort",
]


class LazyScrollsSFTDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        embedding_path: Optional[str] = None,
        dataset_name: str = "qmsum",
        split: str = "train",
        mode: str = "baseline",
        instruction_path: Optional[str] = None,
    ):
        super(LazyScrollsSFTDataset, self).__init__()

        assert dataset_name in scrolls_datasets
        assert split in ["train", "validation"]

        self.instruction_path = instruction_path

        # if instruction data is provided, then use it as our dataset.
        # otherwise, use the scrolls dataset to generate instruction pairs.
        if self.instruction_path is not None:
            rank0_print("Loading instruction data from", instruction_path)
            self.dataset = load_jsonl(instruction_path)
        else:
            rank0_print(f"Loading {dataset_name}_{split}")
            if split == "validation":
                self.dataset = load_dataset("tau/scrolls", dataset_name)["validation"]
                self.dataset = self.__preproc_dataset(self.dataset)
                print("dataset size:", len(self.dataset))
            else:
                self.dataset = load_dataset("tau/scrolls", dataset_name)["train"]

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
        self.is_eval = True if split == "validation" else False
        self.mode = mode
        self.dataset_name = dataset_name

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

    def __process_entry(self, entry):
        if self.instruction_path is not None:
            article_id = entry["article_id"]
            question = entry["question"]
            answer = entry["answer"] + "\nExplanation:" + entry["explanation"]
            context = ""
            return article_id, question, answer, context
        else:
            article_id = entry["id"].split("_")[0]
            input = entry["input"]
            lines = input.splitlines()
            question = lines[0]
            context = "\n".join(lines[1:])
            answer = entry["output"]
            return article_id, question, answer, context

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        entry = self.dataset[i]

        article_id, question, answer, context = self.__process_entry(entry)

        sys_prompt = sys_prompts[self.dataset_name]

        question = sys_prompt + question + "\nAnswer:"

        q_input_ids = self.tokenizer(question, add_special_tokens=False).input_ids
        a_input_ids = self.tokenizer(answer, add_special_tokens=False).input_ids
        a_input_ids += [self.tokenizer.eos_token_id]

        if self.mode == "baseline":
            context = B_SYS + context + truncation_seperator + E_SYS
            c_input_ids = self.tokenizer(
                context,
                padding="longest",
                truncation=True,
                add_special_tokens=False,
                max_length=4000 - len(q_input_ids),
            ).input_ids
            decoder_input_ids = copy.deepcopy(c_input_ids + q_input_ids)
            return {"decoder_input_ids": decoder_input_ids}
        elif self.mode == "baseline_nocontext":
            decoder_input_ids = copy.deepcopy(q_input_ids)
            return {"decoder_input_ids": decoder_input_ids}

        decoder_input_ids = copy.deepcopy(q_input_ids)

        if not self.is_eval:
            decoder_input_ids += a_input_ids
        else:
            print("----------------Question Prompt----------------\n" + question)

        decoder_input_ids = torch.as_tensor(decoder_input_ids)
        labels = copy.deepcopy(decoder_input_ids)
        labels[: len(q_input_ids)] = IGNORE_INDEX

        assert self.is_preprocessed
        context_embeddings = self.context_embeddings_map[article_id]

        if self.is_eval:
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
        return self.dataset[i]["output"]

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
                                        dataset_cls=LazyScrollsSFTDataset,
                                        data_args=data_args,
                                        dataset_name=data_args.dataset_name,
                                        )
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
