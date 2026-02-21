from datasets import load_dataset
from datasets import Dataset
from transformers import AutoTokenizer
import argparse
import time
import multiprocessing
import os
import requests
import json

parser = argparse.ArgumentParser(
    description="Send parallel inference requests to an OpenAI-compatible endpoint."
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=4,
    help="Number of parallel worker processes used to split and send requests (default: 4).",
)
args = parser.parse_args()

if args.batch_size <= 0:
    raise ValueError("--batch-size must be a positive integer")

batch_size = args.batch_size

os.environ["HTTP_PROXY"] = "http://proxy.alcf.anl.gov:3128"
os.environ["HTTPS_PROXY"] = "http://proxy.alcf.anl.gov:3128"
token = ""  # provide your openai token

# Load a dataset with chat messages
ds = load_dataset("lmsys/lmsys-chat-1m", token=token, split="train[0:1483]")


def chat_len(chat):
    return sum([len(msg["content"]) for msg in chat])


inputs = []
outputs = []

for chat in ds["conversation"]:
    input_chat = chat[:-1]
    response = chat[-1]
    in_len = chat_len(input_chat)
    out_len = chat_len([response])
    if 64 <= in_len <= 20000 and 64 <= out_len <= 20000:
        inputs.append(input_chat)
        outputs.append(response)

outputs_ds = Dataset.from_list(outputs)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B", token=token, use_fast=True
)


def len_in_tokens(batch):
    enc = tokenizer(
        batch["content"],
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )
    return {"out_len": [len(ids) for ids in enc["input_ids"]]}


out_lengths_ds = outputs_ds.map(
    len_in_tokens,
    batched=True,
    batch_size=256,
    num_proc=64,
)

out_lengths = out_lengths_ds["out_len"]
print(out_lengths)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
url = "http://localhost:8001/v1/chat/completions"
headers = {"Content-Type": "application/json"}


def make_requests(inputs_slice, out_lengths_slice, output_list):
    for i in range(len(inputs_slice)):
        messages = inputs_slice[i]
        max_tokens = out_lengths_slice[i]

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            output_list.append(json.dumps(response.json(), indent=2))
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")


st = time.time()
with multiprocessing.Manager() as manager:
    shared_outputs = manager.list()
    processes = []

    slice_size = len(inputs) // batch_size
    for i in range(batch_size):
        start_index = i * slice_size
        end_index = (i + 1) * slice_size if i < batch_size - 1 else len(inputs)

        inputs_slice = inputs[start_index:end_index]
        out_lengths_slice = out_lengths[start_index:end_index]

        process = multiprocessing.Process(
            target=make_requests,
            args=(inputs_slice, out_lengths_slice, shared_outputs),
        )

        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    print("".join(shared_outputs))
print(f"time take with BS:{batch_size} = {time.time()-st}")
