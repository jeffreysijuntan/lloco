import io
import json
import torch


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def load_jsonl(f, mode="r"):
    """Load a .jsonl file into a list of dictionaries."""
    f = _make_r_io_base(f, mode)
    jlist = [json.loads(line) for line in f]
    f.close()
    return jlist


def list_to_pt(lst):
    return [torch.as_tensor(x) for x in lst]


def write_to_jsonl(data, filename):
    with open(filename, 'w') as file:
        for entry in data:
            json.dump(entry, file)
            file.write('\n')


def append_to_jsonl(data, filename):
    with open(filename, 'a') as file:  # 'a' mode for appending
        for record in data:
            json_record = json.dumps(record)
            file.write(json_record + '\n')
