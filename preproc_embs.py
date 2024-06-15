import os, random
import fire
import numpy as np
import torch
from auto_compressor import LlocoAutoCompressorModel
from datasets import load_dataset
from needle_util import generate_context
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import load_jsonl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANDOM_NEEDLE_CITIES  = [
        'Chicago', 'Yangon', 'Antananarivo', 'Colombo', 'Almaty', 'Sydney', 'Chicago', 'Mexico City',
        'Seattle', 'Lagos', 'Amsterdam', 'Belgrade', 'Cairo', 'Baghdad', 'Damascus', 'Kigali', 'Dakar',
        'Dakar', 'Sofia', 'Kigali', 'Victoria', 'Tashkent', 'Mumbai', 'Barcelona', 'Almaty', 'Amman',
        'Toronto', 'Bratislava', 'Johannesburg', 'Thimphu', 'Bangkok', 'Santiago', 'Cairo', 'San Francisco',
        'Lagos', 'Amsterdam', 'Paris', 'Rabat', 'Santiago', 'Copenhagen', 'Madrid', 'Kigali',
        'Ho Chi Minh City', 'Sarajevo', 'Delhi', 'Istanbul', 'Ho Chi Minh City', 'Khartoum', 'Helsinki',
]


RANDOM_TEST_NEEDLE_CITIES = [
        'Doha', 'Istanbul', 'Kuala Lumpur', 'Budapest', 'Shanghai', 'Moscow', 'Los Angeles', 'Oslo',
        'Johannesburg', 'Berlin', 'Bangalore', 'Tokyo', 'Melbourne', 'Barcelona', 'Chicago', 'Port Louis',
        'Lisbon', 'Nairobi', 'Kampala', 'Lima', 'Maputo', 'Vancouver', 'Dubai', 'Khartoum', 'Jakarta',
        'Madrid', 'Yerevan', 'Beirut', 'Athens', 'Chicago', 'Paris', 'Bucharest', 'Copenhagen', 'Brussels',
        'Damascus', 'Seattle', 'Los Angeles', 'Yerevan', 'Victoria', 'Tunis', 'Astana', 'Seoul',
        'Buenos Aires', 'Bangkok', 'Colombo', 'Brussels', 'Khartoum', 'Doha', 'San Francisco', 'Vienna', 'Jakarta'
]

def generate_random_number(num_digits):
    lower_bound = 10**(num_digits - 1)
    upper_bound = 10**num_digits - 1
    return random.randint(lower_bound, upper_bound)


class EmbeddingPreprocessor(object):
    def __init__(self, emb_model_name="autocomp", max_length=6144, chunk_size=16, icae_checkpoint=None, truncation=True):
        self.emb_model_name = emb_model_name

        if self.emb_model_name == "autocomp":
            self.tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/AutoCompressor-Llama-2-7b-6k")
            self.emb_model = LlocoAutoCompressorModel.from_pretrained("princeton-nlp/AutoCompressor-Llama-2-7b-6k", torch_dtype=torch.bfloat16).eval().cuda()
            self.tokenizer.pad_token = '[PAD]'
        else:
            raise NotImplementedError

        self.max_len = max_length
        self.chunk_len = chunk_size
        self.truncation = truncation

    def preprocess_single(self, context):
        if self.emb_model_name == "autocomp":
            return self.preprocess_context_autocomp(context)
        else:
            raise NotImplementedError
        

    @torch.inference_mode()
    def preprocess_batch(self, contexts, batch_size=8):
        assert self.emb_model_name == "autocomp"
        for i in range(0, len(contexts), batch_size):
            batch = contexts[i:i+batch_size]
            if self.truncation:
                context_tokens = self.tokenizer(batch, padding=True, add_special_tokens=False, return_tensors="pt", max_length=6144, truncation=True).input_ids.cuda()
            else:
                context_tokens = self.tokenizer(batch, padding=True, add_special_tokens=False, return_tensors="pt", truncation=False).input_ids.cuda()
            summary_vectors = self.emb_model(context_tokens, segment_lengths=1536, output_softprompt=True).softprompt
            print("summary_vectors", summary_vectors.shape)
            return summary_vectors.to("cpu")
        

    @torch.inference_mode()
    def preprocess_context_autocomp(self, context):
        if not self.truncation and len(context) > 10000:
            context_tokens = []
            for i in range(0, len(context), 10000):
                ctx = context[i:i+10000]
                ctx_tokens = self.tokenizer(ctx, add_special_tokens=False, return_tensors="pt", truncation=False).input_ids
                context_tokens += ctx_tokens
            context_tokens = torch.cat(context_tokens)
            context_tokens = context_tokens[:122880].unsqueeze_(dim=0).cuda()
        else:
            if self.truncation:
                context_tokens = self.tokenizer(context, add_special_tokens=False, return_tensors="pt", max_length=6144, truncation=True).input_ids.cuda()
            else:
                context_tokens = self.tokenizer(context, add_special_tokens=False, max_length=120000, return_tensors="pt", truncation=False).input_ids.cuda()

        print("context_tokens", context_tokens.shape)
        summary_vectors = self.emb_model(context_tokens.long(), segment_lengths=1536, output_softprompt=True).softprompt
        print("summary_vectors", summary_vectors.shape)
        return summary_vectors[0].to("cpu")


def preprocess_quality(preprocessor, quality_path, out_path):
    dataset = load_jsonl(quality_path)
    cache = {}
    visited = set()
    for entry in tqdm(dataset):
        pid = entry["article_id"]
        if pid in visited:
            continue
        visited.add(pid)

        context = entry["article"]
        emb = preprocessor.preprocess_single(context)
        cache[pid] = emb

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(cache, out_path)


def preprocess_qasper(preprocessor, split="train", out_path=None):
    assert split in ["train", "validation"]
    dataset = load_dataset("tau/scrolls", "qasper")[split]

    cache = {}
    hash_id_map = {}
    for entry in tqdm(dataset):
        pid = entry["id"]
        input = entry["input"]
        lines = input.splitlines()
        context = "\n".join(lines[1:])

        ctx_hash = hash(context)
        if ctx_hash in hash_id_map:
            first_id = hash_id_map[ctx_hash]
            cache[pid] = cache[first_id] # embedding already in cache, so just take it.
            continue
        hash_id_map[ctx_hash] = pid
        emb = preprocessor.preprocess_single(context)
        cache[pid] = emb

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(cache, out_path)


def preprocess_scrolls(preprocessor, dataset_name, split="train", out_path=None):
    assert split in ["train", "validation"]
    dataset = load_dataset("tau/scrolls", dataset_name)[split]

    cache = {}
    hash_id_map = {}
    for entry in tqdm(dataset):
        pid = entry["id"]
        pid = pid.split("_")[0]

        input = entry["input"]
        lines = input.splitlines()
        context = "\n".join(lines[1:])

        ctx_hash = hash(context)
        if ctx_hash in hash_id_map:
            first_id = hash_id_map[ctx_hash]
            cache[pid] = cache[first_id]
            continue
        hash_id_map[ctx_hash] = pid
        emb = preprocessor.preprocess_single(context)
        cache[pid] = emb

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(cache, out_path)


def preprocess_hotpotqa(preprocessor, split="train", out_path=None, n_sample=20000):
    dataset = load_dataset("hotpot_qa", "fullwiki", split=split)
    if split == "train":
        print(f"Shuffling and selecting {n_sample} examples from the training set...")
        dataset = dataset.shuffle(seed=42).select(range(n_sample))
    cache = {}
    for entry in tqdm(dataset):
        pid = entry["id"]
        
        context = ""
        for i, (title, sentences) in enumerate(
            zip(entry["context"]['title'], entry["context"]['sentences'])):
            if i > 0:
                context += '\n\n'
            context += title + '\n'
            for sent in sentences:
                context += sent

        emb = preprocessor.preprocess_single(context)
        cache[pid] = emb

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(cache, out_path)


def preprocess_needle(preprocessor, tokenizer, needle, context_length, out_path, context):
    context_lengths = np.round(np.linspace(1000, context_length, num=10, endpoint=True)).astype(int)
    document_depth_percents = np.round(np.linspace(0, 100, num=10, endpoint=True)).astype(int)
    cache = {}
    for ctx_len in tqdm(context_lengths):
        for depth_percent in tqdm(document_depth_percents, leave=False):
            content = generate_context(tokenizer, needle, context, ctx_len, depth_percent)
            emb = preprocessor.preprocess_single(tokenizer.decode(content))
            key = f"{ctx_len}_{depth_percent}"
            cache[key] = emb
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(cache, out_path)


def preprocess_needle_magic_city(preprocessor, tokenizer, context_length, out_path, context):
    context_lengths = np.round(np.linspace(1000, context_length, num=10, endpoint=True)).astype(int)
    document_depth_percents = np.round(np.linspace(0, 100, num=10, endpoint=True)).astype(int)
    cache = {}

    for iter in range(10):
        for ctx_len in tqdm(context_lengths):
            for depth_percent in tqdm(document_depth_percents, leave=False):
                rnd_needle = random.choice(RANDOM_TEST_NEEDLE_CITIES)
                needle = f"The magic city is {rnd_needle}"
                print(needle)
                content = generate_context(tokenizer, needle, context, ctx_len, depth_percent)
                emb = preprocessor.preprocess_single(tokenizer.decode(content))
                key = f"{ctx_len}_{depth_percent}_{iter}"
                cache[key] = emb
                cache[f"{key}_needle"] = rnd_needle

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(cache, out_path)


def main(
        emb_model_name, dataset, split="train", truncation=True, data_path=None, 
        out_path=None, n_sample=20000, 
        model_name_or_path="meta-llama/Llama-2-7b-chat-hf", context_length=32000):
    preprocessor = EmbeddingPreprocessor(emb_model_name, truncation=truncation)
    if dataset == "quality":
        preprocess_quality(preprocessor, data_path, out_path)
    elif dataset == "hotpot_qa":
        preprocess_hotpotqa(preprocessor, split=split, out_path=out_path, n_sample=n_sample)
    elif dataset in ["qasper", "narrative_qa", "qmsum"]:
        preprocess_scrolls(preprocessor, dataset, split=split, out_path=out_path)
    elif dataset == "needle":
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        needle = "Mary's favourite fashion designer was Coco Chanel when she was a teenager."
        dataset = load_dataset("tau/scrolls", "narrative_qa")[split]
        input = dataset[97]["input"]
        lines = input.splitlines()
        context = "\n".join(lines[1:])
        preprocess_needle(preprocessor, tokenizer, needle, context_length, out_path=out_path, context=context)
    elif dataset == "needle_magic_city":
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        dataset = load_dataset("tau/scrolls", "narrative_qa")[split]
        input = dataset[97]["input"]
        lines = input.splitlines()
        context = "\n".join(lines[1:])
        preprocess_needle_magic_city(preprocessor, tokenizer, context_length, out_path=out_path, context=context)
    else:
        raise NotImplementedError("Dataset not supported.")


if __name__ == "__main__":
    fire.Fire(main)