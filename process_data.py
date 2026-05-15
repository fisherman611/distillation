import multiprocessing
import os
import time
import torch
import json
import sys
from numerize.numerize import numerize
import numpy as np
from data_utils.indexed_dataset import make_builder
from transformers import AutoTokenizer
from arguments import get_args


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path,
            padding_side="right"
        )

    @staticmethod
    def _apply_chat_template(messages, add_generation_prompt):
        # Some tokenizers (e.g., Qwen) support enable_thinking, others (e.g., Llama) do not.
        try:
            return Encoder.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            return Encoder.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            )

    def encode(self, line):
        line = json.loads(line)

        system_prompt = line["system_prompt"]
        user_prompt = line["user_prompt"]
        response = line.get("response", "")

        t_system_prompt = line.get("t_system_prompt")
        t_user_prompt = line.get("t_user_prompt")

        prompt = self._apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            add_generation_prompt=True,
        )

        full_prompt_tokens = Encoder.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_tokens = full_prompt_tokens

        if len(prompt_tokens) > self.args.max_prompt_length:
            prompt_tokens = prompt_tokens[:self.args.max_prompt_length]

        response_tokens = None
        if self.args.split in ["train", "valid"]:
            full_tokens = Encoder.tokenizer.encode(
                prompt + response,
                add_special_tokens=False,
            ) + [Encoder.tokenizer.eos_token_id]
            # Split response using untruncated prompt length to avoid leaking prompt tail into labels.
            response_tokens = full_tokens[len(full_prompt_tokens):]
            if not response_tokens or response_tokens[-1] != Encoder.tokenizer.eos_token_id:
                response_tokens = (response_tokens or []) + [Encoder.tokenizer.eos_token_id]

        t_prompt_tokens = None
        if self.args.split == "train" and t_system_prompt is not None and t_user_prompt is not None:
            t_prompt = self._apply_chat_template(
                [
                    {"role": "system", "content": t_system_prompt},
                    {"role": "user", "content": t_user_prompt},
                ],
                add_generation_prompt=True,
            )

            t_prompt_tokens = Encoder.tokenizer.encode(t_prompt, add_special_tokens=False)
            if len(t_prompt_tokens) > self.args.t_max_prompt_length:
                t_prompt_tokens = t_prompt_tokens[:self.args.t_max_prompt_length]

        return line, prompt, prompt_tokens, response_tokens, t_prompt_tokens, len(line)


def load_split_data(args):
    split_to_file = {
        "train": "train.jsonl",
        "valid": "dev.jsonl",
        "test": "test.jsonl",
    }

    if args.split not in split_to_file:
        raise ValueError(f"Unsupported split: {args.split}")

    file_path = os.path.join(args.data_dir, split_to_file[args.split])
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find file for split '{args.split}': {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = f.readlines()

    return data


def get_builder_dtype(tokenizer):
    # Use uint16 only when vocab is safely below separator value 65535.
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    return np.int32


def main():
    print("OK")
    args = get_args()

    if args.split is None:
        raise ValueError("Please provide --split with one of: train, valid, test")

    if "generated" not in args.processed_data_dir:
        args.processed_data_dir = os.path.join(args.processed_data_dir, args.model_type)

    os.makedirs(args.processed_data_dir, exist_ok=True)

    data = load_split_data(args)

    encoder = Encoder(args)

    pool = multiprocessing.Pool(
        processes=args.data_process_workers,
        initializer=encoder.initializer
    )
    encoded_docs = pool.imap_unordered(encoder.encode, data, chunksize=50)

    proc_start = time.time()
    total_bytes_processed = 0

    bin_file = os.path.join(args.processed_data_dir, f"{args.split}_0.bin")
    idx_file = os.path.join(args.processed_data_dir, f"{args.split}_0.idx")

    tokenizer_for_dtype = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side="right",
    )
    dtype = get_builder_dtype(tokenizer_for_dtype)
    binary_builder = make_builder(bin_file, impl="mmap", dtype=dtype)

    t_binary_builder = None
    t_bin_file = None
    t_idx_file = None
    if args.split == "train":
        t_bin_file = os.path.join(args.processed_data_dir, "teacher_train_0.bin")
        t_idx_file = os.path.join(args.processed_data_dir, "teacher_train_0.idx")

    inst_num = 0
    print("#" * 10, args.split, "#" * 10)

    prompt_lens = []
    response_lens = []

    json_file = open(
        os.path.join(args.processed_data_dir, f"{args.split}.jsonl"),
        "w",
        encoding="utf-8"
    )

    for lid, (line, prompt_str, prompt, response, t_prompt, bytes_processed) in enumerate(encoded_docs):
        total_bytes_processed += bytes_processed

        if prompt is None:
            continue

        if args.split in ["train", "valid"]:
            if response is None:
                continue

            if args.only_prompt:
                if len(prompt) < args.max_length:
                    binary_builder.add_item(torch.IntTensor(prompt))
                else:
                    continue
            else:
                binary_builder.add_item(torch.IntTensor(prompt + [-1] + response))

                if args.split == "train" and t_prompt is not None:
                    if t_binary_builder is None:
                        t_binary_builder = make_builder(t_bin_file, impl="mmap", dtype=dtype)
                    t_binary_builder.add_item(torch.IntTensor(t_prompt + [-1] + response))

            response_lens.append(len(response))

        else:  # test
            if len(prompt) < args.max_length:
                binary_builder.add_item(torch.IntTensor(prompt))
            else:
                continue

        json_file.write(
            json.dumps(
                {
                    "prompt": prompt_str,
                    "response": line.get("response", ""),
                },
                ensure_ascii=False,
            ) + "\n"
        )

        prompt_lens.append(len(prompt))
        inst_num += 1

        if lid % 1000 == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024 if elapsed > 0 else 0
            print(
                f"Processed {lid} documents. {inst_num} instances. "
                f"({lid / elapsed if elapsed > 0 else 0:.2f} docs/s, {mbs:.2f} MB/s).",
                file=sys.stderr,
            )

    binary_builder.finalize(idx_file)
    if t_binary_builder is not None:
        t_binary_builder.finalize(t_idx_file)

    pool.close()
    pool.join()
    json_file.close()

    print("Data num", len(prompt_lens))
    if len(prompt_lens) > 0:
        print(
            "Prompt lengths.",
            "Mean:", np.mean(prompt_lens),
            "Max:", np.max(prompt_lens),
            "Min:", np.min(prompt_lens),
        )
    if len(response_lens) > 0:
        print(
            "Response lengths.",
            "Mean:", np.mean(response_lens),
            "Max:", np.max(response_lens),
            "Min:", np.min(response_lens),
        )


if __name__ == '__main__':
    main()
