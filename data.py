from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers

IGNORE_INDEX = -100


@dataclass
class DataCollatorForLLoCOSFTDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        assert "inputs_embeds" in instances[0]
        input_ids, labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels")
        )
        context_embeddings = None
        if "inputs_embeds" in instances[0]:
            context_embeddings = torch.stack(
                [instance["inputs_embeds"] for instance in instances]
            )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        ret = dict(
            input_ids=input_ids,
            labels=labels,
            inputs_embeds=context_embeddings,
            segment_lengths=9999,
            output_hidden_states=True,
        )
        return ret


def make_lloco_data_module(model, tokenizer, dataset_cls, data_args, **kwargs) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if not data_args.lazy_preprocess:
        raise NotImplementedError
    else:
        train_dataset = dataset_cls(
            tokenizer=tokenizer,
            embedding_path=data_args.embedding_path,
            split="train",
            mode=data_args.eval_mode,
            **kwargs,
        )
    print("Dataset size:", len(train_dataset))
    data_collator = DataCollatorForLLoCOSFTDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )