from datasets import load_dataset
from datasets import Dataset
from transformers import AutoTokenizer
import pdb

import multiprocessing

# Load a dataset with chat messages
ds = load_dataset("lmsys/lmsys-chat-1m", token='', split="train[0:1483]")

def chat_len(chat):
    return sum([len(msg['content']) for msg in chat])
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
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", token='', use_fast=True)

def len_in_tokens(batch):
    enc = tokenizer(
        batch["content"],
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )
    # enc["input_ids"] is now a list-of-lists, one per example in the batch.
    # We need to return a list of lengths, not just a single int.
    return {"out_len": [len(ids) for ids in enc["input_ids"]]}

out_lengths_ds = outputs_ds.map(
    len_in_tokens,
    batched=True,
    batch_size=256,
    num_proc=64
)

out_lengths = out_lengths_ds["out_len"]
print(out_lengths)
import requests
import json
url = "http://localhost:8001/v1/chat/completions"
headers = {"Content-Type": "application/json"}

def make_requests(inputs_slice, out_lengths_slice, output_list):
    for i in range(len(inputs_slice)):
        messages = inputs_slice[i]
        max_tokens = out_lengths_slice[i]
        
        payload = {
            "messages": messages,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status() 
            output_list.append(json.dumps(response.json(), indent=2))
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            
batch_size = 8

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
            args=(inputs_slice, out_lengths_slice, shared_outputs)
        )
        
        processes.append(process)
        process.start()
    
    for process in processes:
        process.join()

    print("".join(shared_outputs))