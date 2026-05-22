import math
import os
import re
import random
import time

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

try:
    from ._bootstrap import configure_project_paths
except ImportError:
    from _bootstrap import configure_project_paths

configure_project_paths()

from data_utils.lm_datasets import LMTrainDataset
from data_utils import lm_datasets
from arguments import get_args
from distillm import ReplayBuffer, SampleGenerator
try:
    from .span_finetune import (
        prepare_dataset as base_prepare_dataset,
        evaluate,
        get_distil_loss,
        get_teacher_model,
        pt_loss,
        setup_model_and_optimizer,
    )
except ImportError:
    from span_finetune import (
        prepare_dataset as base_prepare_dataset,
        evaluate,
        get_distil_loss,
        get_teacher_model,
        pt_loss,
        setup_model_and_optimizer,
    )
from utils import get_tokenizer, initialize, print_args, print_rank, save_rank


torch.set_num_threads(4)


QUESTION_MARKER_TEXT = "Question:\n"
SCHEMA_MARKER_TEXT = "Schema:\n"
SCHEMA_END_MARKER_TEXT = (
    "\n\nFirst, make a plan.\n"
    "Then, execute the plan step by step.\n"
    "Finally, give the answer."
)
SCHEMA_LINKING_OPEN_TAGS = (
    "<schema_linking>",
    "\n<schema_linking>",
    "\n\n<schema_linking>",
    "<SCHEMA_LINKING>",
    "\n<SCHEMA_LINKING>",
    "\n\n<SCHEMA_LINKING>",
)
SCHEMA_LINKING_CLOSE_TAGS = (
    "</schema_linking>",
    "\n</schema_linking>",
    "\n\n</schema_linking>",
    "</SCHEMA_LINKING>",
    "\n</SCHEMA_LINKING>",
    "\n\n</SCHEMA_LINKING>",
)


FINAL_CYPHER_TAG_PATTERN = re.compile(
    r"<final_cypher>(.*?)</final_cypher>",
    flags=re.IGNORECASE | re.DOTALL,
)
SCHEMA_LINKING_TAG_PATTERN = re.compile(
    r"<schema_linking>(.*?)</schema_linking>",
    flags=re.IGNORECASE | re.DOTALL,
)
_TOKENIZED_MARKER_CACHE = {}
SCHEMA_LINKING_SPAN_TYPE = "schema_linking"


def _zero_loss(device):
    return torch.zeros((), dtype=torch.float32, device=device)


def _extract_tag_content_with_offsets(text, pattern):
    if not isinstance(text, str):
        return None, -1, -1

    match = pattern.search(text)
    if match is None:
        return None, -1, -1

    inner_text = match.group(1)
    start = match.start(1)

    left_trim = len(inner_text) - len(inner_text.lstrip())
    right_trim = len(inner_text.rstrip())
    if right_trim <= left_trim:
        return None, -1, -1

    content_start = start + left_trim
    content_end = start + right_trim
    content = text[content_start:content_end]
    if not content.strip():
        return None, -1, -1
    return content, content_start, content_end


def extract_text2cypher_span_items_from_response_new_data(response_str):
    cypher_query, cypher_start, _ = _extract_tag_content_with_offsets(
        response_str,
        FINAL_CYPHER_TAG_PATTERN,
    )
    if cypher_query is None:
        return []

    span_items = lm_datasets.extract_text2cypher_span_items(cypher_query)
    for item in span_items:
        item["start"] += cypher_start
        item["end"] += cypher_start

    schema_linking, schema_link_start, schema_link_end = _extract_tag_content_with_offsets(
        response_str,
        SCHEMA_LINKING_TAG_PATTERN,
    )
    if schema_linking is not None:
        span_items.append(
            {
                "type": "schema_linking",
                "start": schema_link_start,
                "end": schema_link_end,
                "text": schema_linking,
            }
        )

    unique = {}
    for item in span_items:
        unique[(item["type"], item["start"], item["end"])] = item
    deduped_items = list(unique.values())
    deduped_items.sort(key=lambda x: (x["start"], x["end"], x["type"]))
    return deduped_items


def extract_text2cypher_span_offsets_new_data(full_text, response_str):
    response_start = lm_datasets._find_response_start(full_text, response_str)
    span_items = extract_text2cypher_span_items_from_response_new_data(response_str)

    unique = {}
    for item in span_items:
        start = response_start + item["start"]
        end = response_start + item["end"]
        if start >= end:
            continue
        span_type = item.get("type")
        unique[(span_type, start, end)] = {
            "type": span_type,
            "start": start,
            "end": end,
        }

    return [unique[key] for key in sorted(unique, key=lambda x: (x[1], x[2], x[0] or ""))]


def patch_response_span_extractor_for_new_data():
    lm_datasets.extract_text2cypher_span_items_from_response = (
        extract_text2cypher_span_items_from_response_new_data
    )
    lm_datasets.extract_text2cypher_span_offsets = extract_text2cypher_span_offsets_new_data


def prepare_dataset_new_data_safe(args, tokenizer):
    """
    New reasoning data on HF may not include test mmap files (e.g. test_0.idx).
    For training-only runs, gracefully skip test split instead of crashing.
    """
    try:
        return base_prepare_dataset(args, tokenizer)
    except FileNotFoundError as exc:
        missing_test = ("test_0.idx" in str(exc)) or ("test.idx" in str(exc))
        if not (missing_test and args.do_train and not args.do_eval):
            raise

        print_rank("test split not found for current data_dir; continue with train/dev only.")
        data = {}
        rng_sample = random.Random(args.seed)

        data["train"] = LMTrainDataset(
            args,
            tokenizer,
            args.data_dir,
            "train",
            args.train_num,
            args.train_ratio,
            rng_sample,
        )
        print_rank("train num", len(data["train"]))
        data["dev"] = LMTrainDataset(
            args,
            tokenizer,
            args.data_dir,
            "valid",
            args.dev_num,
            args.dev_ratio,
            rng_sample,
        )

        if args.lm_data_dir is not None:
            data["pt_train"] = LMTrainDataset(
                args,
                tokenizer,
                args.lm_data_dir,
                "train",
                args.train_num,
                args.train_ratio,
                rng_sample,
            )
            print_rank("train num", len(data["pt_train"]))

        return data


def get_grounding_loss_config(args):
    w_rel = float(getattr(args, "w_rel_loss", 0.0))
    if w_rel == 0.0:
        w_rel = float(getattr(args, "w_span_loss", 1.0))
    if (not math.isfinite(w_rel)) or w_rel < 0.0:
        w_rel = 1.0

    w_span_query = getattr(args, "w_span_query_loss", None)
    w_span_schema = getattr(args, "w_span_schema_loss", None)
    w_span_schema_linking = getattr(args, "w_span_schema_linking_loss", None)

    if w_span_query is None and w_span_schema is None and w_span_schema_linking is None:
        w_span_query = w_rel / 3.0
        w_span_schema = w_rel / 3.0
        w_span_schema_linking = w_rel / 3.0
    else:
        w_span_query = 0.0 if w_span_query is None else w_span_query
        w_span_schema = 0.0 if w_span_schema is None else w_span_schema
        w_span_schema_linking = 0.0 if w_span_schema_linking is None else w_span_schema_linking

    w_span_query = float(w_span_query)
    w_span_schema = float(w_span_schema)
    w_span_schema_linking = float(w_span_schema_linking)

    if (not math.isfinite(w_span_query)) or w_span_query < 0.0:
        w_span_query = 0.0
    if (not math.isfinite(w_span_schema)) or w_span_schema < 0.0:
        w_span_schema = 0.0
    if (not math.isfinite(w_span_schema_linking)) or w_span_schema_linking < 0.0:
        w_span_schema_linking = 0.0

    return {
        "w_span_query": min(w_span_query, 1e4),
        "w_span_schema": min(w_span_schema, 1e4),
        "w_span_schema_linking": min(w_span_schema_linking, 1e4),
    }


def build_prompt_token_mask(attention_mask, labels):
    valid_token_mask = attention_mask.bool()
    prompt_mask = (labels == -100) & valid_token_mask
    no_prompt = (~prompt_mask.any(dim=-1)) & valid_token_mask.any(dim=-1)
    if no_prompt.any():
        prompt_mask[no_prompt] = valid_token_mask[no_prompt]
    return prompt_mask


def build_generated_no_model_batch(labels):
    return {
        "label": labels,
        "loss_mask": (labels != -100).float(),
    }


def _tokenize_marker(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def _get_tokenized_markers(tokenizer):
    cache_key = id(tokenizer)
    markers = _TOKENIZED_MARKER_CACHE.get(cache_key)
    if markers is None:
        markers = {
            "question": _tokenize_marker(tokenizer, QUESTION_MARKER_TEXT),
            "schema": _tokenize_marker(tokenizer, SCHEMA_MARKER_TEXT),
            "schema_end": _tokenize_marker(tokenizer, SCHEMA_END_MARKER_TEXT),
            "schema_linking_open": [
                _tokenize_marker(tokenizer, marker) for marker in SCHEMA_LINKING_OPEN_TAGS
            ],
            "schema_linking_close": [
                _tokenize_marker(tokenizer, marker) for marker in SCHEMA_LINKING_CLOSE_TAGS
            ],
        }
        _TOKENIZED_MARKER_CACHE[cache_key] = markers
    return markers


def _find_subsequence(sequence, pattern, start=0):
    if not pattern:
        return -1, 0

    start = max(start, 0)
    pattern_len = len(pattern)
    max_start = len(sequence) - pattern_len
    if start > max_start:
        return -1, 0

    first_token = pattern[0]
    for idx in range(start, max_start + 1):
        if sequence[idx] == first_token and sequence[idx : idx + pattern_len] == pattern:
            return idx, len(pattern)
    return -1, 0


def _find_first_subsequence(sequence, patterns, start=0):
    best_idx = -1
    best_len = 0
    for pattern in patterns:
        idx, length = _find_subsequence(sequence, pattern, start=start)
        if idx >= 0 and (best_idx < 0 or idx < best_idx):
            best_idx = idx
            best_len = length
    return best_idx, best_len


def build_prompt_section_masks(input_ids, attention_mask, labels, tokenizer):
    prompt_mask = build_prompt_token_mask(attention_mask, labels)
    query_mask = torch.zeros_like(prompt_mask)
    schema_mask = torch.zeros_like(prompt_mask)

    markers = _get_tokenized_markers(tokenizer)
    question_marker = markers["question"]
    schema_marker = markers["schema"]
    schema_end_marker = markers["schema_end"]

    for batch_idx in range(input_ids.size(0)):
        prompt_indices = torch.nonzero(prompt_mask[batch_idx], as_tuple=False).flatten()
        if prompt_indices.numel() == 0:
            continue

        token_ids = input_ids[batch_idx, prompt_indices].detach().cpu().tolist()
        question_pos, question_len = _find_subsequence(token_ids, question_marker)
        schema_pos, schema_len = _find_subsequence(
            token_ids,
            schema_marker,
            start=max(question_pos + question_len, 0),
        )
        if question_pos < 0 or schema_pos < 0:
            continue

        query_start = question_pos + question_len
        query_end = schema_pos
        schema_start = schema_pos + schema_len
        schema_end, _ = _find_subsequence(token_ids, schema_end_marker, start=schema_start)
        if schema_end < 0:
            schema_end = len(token_ids)

        if query_start < query_end:
            query_mask[batch_idx, prompt_indices[query_start:query_end]] = True
        if schema_start < schema_end:
            schema_mask[batch_idx, prompt_indices[schema_start:schema_end]] = True

    query_mask = query_mask & attention_mask.bool()
    schema_mask = schema_mask & attention_mask.bool()
    return query_mask, schema_mask


def build_response_schema_linking_mask(input_ids, attention_mask, labels, tokenizer):
    response_mask = (labels != -100) & attention_mask.bool()
    schema_linking_mask = torch.zeros_like(response_mask)

    markers = _get_tokenized_markers(tokenizer)
    open_markers = markers["schema_linking_open"]
    close_markers = markers["schema_linking_close"]

    for batch_idx in range(input_ids.size(0)):
        response_indices = torch.nonzero(response_mask[batch_idx], as_tuple=False).flatten()
        if response_indices.numel() == 0:
            continue

        token_ids = input_ids[batch_idx, response_indices].detach().cpu().tolist()
        open_pos, open_len = _find_first_subsequence(token_ids, open_markers)
        if open_pos < 0:
            continue

        content_start = open_pos + open_len
        close_pos, _ = _find_first_subsequence(token_ids, close_markers, start=content_start)
        if close_pos < 0:
            continue

        if content_start < close_pos:
            schema_linking_mask[batch_idx, response_indices[content_start:close_pos]] = True

    schema_linking_mask = schema_linking_mask & attention_mask.bool()
    return schema_linking_mask


def prepare_span_token_map(attention_mask, offsets_mapping, spans_offsets):
    device = attention_mask.device
    batch_size, seq_len = attention_mask.shape

    normalized_spans_offsets = []
    span_types = []
    for sample_spans in spans_offsets:
        sample_offsets = []
        sample_types = []
        for span in sample_spans:
            if isinstance(span, dict):
                start = span.get("start")
                end = span.get("end")
                span_type = span.get("type")
            else:
                start, end = span[:2]
                span_type = None

            if start is None or end is None:
                continue
            sample_offsets.append((int(start), int(end)))
            sample_types.append(span_type)
        normalized_spans_offsets.append(sample_offsets)
        span_types.append(sample_types)

    max_spans = max((len(sample_spans) for sample_spans in normalized_spans_offsets), default=0)
    if max_spans == 0:
        return None, None, None

    span_starts = torch.zeros(batch_size, max_spans, dtype=torch.long, device=device)
    span_ends = torch.zeros(batch_size, max_spans, dtype=torch.long, device=device)
    span_mask = torch.zeros(batch_size, max_spans, dtype=torch.bool, device=device)
    schema_linking_span_mask = torch.zeros(batch_size, max_spans, dtype=torch.bool, device=device)

    for batch_idx, sample_spans in enumerate(normalized_spans_offsets):
        if not sample_spans:
            continue
        spans_tensor = torch.tensor(sample_spans, dtype=torch.long, device=device)
        span_starts[batch_idx, : len(sample_spans)] = spans_tensor[:, 0]
        span_ends[batch_idx, : len(sample_spans)] = spans_tensor[:, 1]
        span_mask[batch_idx, : len(sample_spans)] = True
        if span_types[batch_idx]:
            schema_linking_span_mask[batch_idx, : len(sample_spans)] = torch.tensor(
                [span_type == SCHEMA_LINKING_SPAN_TYPE for span_type in span_types[batch_idx]],
                dtype=torch.bool,
                device=device,
            )

    current_offsets = offsets_mapping[:, :seq_len, :] if offsets_mapping.shape[1] != seq_len else offsets_mapping
    token_start = current_offsets[..., 0].unsqueeze(-1).to(device)
    token_end = current_offsets[..., 1].unsqueeze(-1).to(device)

    token_in_span = (token_start + 1 >= span_starts.unsqueeze(1)) & (token_end <= span_ends.unsqueeze(1))
    token_in_span = token_in_span & attention_mask.unsqueeze(-1).bool() & span_mask.unsqueeze(1)

    if not token_in_span.any():
        return None, None, None

    cypher_span_mask = span_mask & ~schema_linking_span_mask
    schema_linking_token_mask = (token_in_span & schema_linking_span_mask.unsqueeze(1)).any(dim=-1)
    return token_in_span, cypher_span_mask, schema_linking_token_mask


def _safe_cosine_similarity(x, y, dim=-1, eps=1e-6):
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    y = torch.nan_to_num(y.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    x = F.normalize(x, p=2, dim=dim, eps=eps)
    y = F.normalize(y, p=2, dim=dim, eps=eps)
    return (x * y).sum(dim=dim).clamp(min=-1.0, max=1.0)


def compute_sample_cypher_span_representations(hidden_state, token_to_span_map):
    token_weights = token_to_span_map.float()
    hidden_f = torch.nan_to_num(hidden_state.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    span_sums = torch.einsum("ld,ls->sd", hidden_f, token_weights)
    span_lengths = token_weights.sum(dim=0).unsqueeze(-1).clamp(min=1e-5)
    return span_sums / span_lengths


def compute_section_attention_representations(hidden_state, cypher_spans, section_mask):
    token_positions = torch.nonzero(section_mask, as_tuple=False).flatten()
    return compute_section_attention_representations_from_positions(
        hidden_state,
        cypher_spans,
        token_positions,
    )


def compute_section_attention_representations_from_positions(hidden_state, cypher_spans, token_positions):
    if token_positions.numel() == 0 or cypher_spans.size(0) == 0:
        return None

    section_hidden = hidden_state[token_positions].float()
    section_hidden = torch.nan_to_num(section_hidden, nan=0.0, posinf=1e4, neginf=-1e4)
    cypher_spans = torch.nan_to_num(cypher_spans.float(), nan=0.0, posinf=1e4, neginf=-1e4)

    scores = torch.matmul(cypher_spans, section_hidden.transpose(0, 1))
    scores = scores / math.sqrt(hidden_state.size(-1))
    scores = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)
    scores = scores.clamp(min=-1e4, max=1e4)

    attn_weights = torch.softmax(scores, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
    attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-5)

    result = torch.matmul(attn_weights, section_hidden)
    return torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)


def compute_aligned_span_relation_loss_for_section(
    student_hidden_state,
    teacher_hidden_state,
    token_to_span_map,
    span_mask,
    section_mask,
):
    return compute_aligned_span_relation_losses_for_sections(
        student_hidden_state,
        teacher_hidden_state,
        token_to_span_map,
        span_mask,
        (section_mask,),
    )[0]


def compute_aligned_span_relation_losses_for_sections(
    student_hidden_state,
    teacher_hidden_state,
    token_to_span_map,
    span_mask,
    section_masks,
):
    zero = _zero_loss(student_hidden_state.device)
    loss_nums = [zero for _ in section_masks]
    loss_dens = [zero for _ in section_masks]

    for batch_idx in range(student_hidden_state.size(0)):
        section_positions = [
            torch.nonzero(section_mask[batch_idx], as_tuple=False).flatten()
            for section_mask in section_masks
        ]
        if not any(positions.numel() > 0 for positions in section_positions):
            continue

        cypher_span_lengths = token_to_span_map[batch_idx].float().sum(dim=0)
        valid_span_mask = span_mask[batch_idx] & (cypher_span_lengths > 0)
        if not valid_span_mask.any():
            continue

        student_cypher_spans = compute_sample_cypher_span_representations(
            student_hidden_state[batch_idx],
            token_to_span_map[batch_idx],
        )[valid_span_mask]
        teacher_cypher_spans = compute_sample_cypher_span_representations(
            teacher_hidden_state[batch_idx],
            token_to_span_map[batch_idx],
        )[valid_span_mask]
        weights = cypher_span_lengths[valid_span_mask]

        for section_idx, token_positions in enumerate(section_positions):
            if token_positions.numel() == 0:
                continue

            student_section_spans = compute_section_attention_representations_from_positions(
                student_hidden_state[batch_idx],
                student_cypher_spans,
                token_positions,
            )
            teacher_section_spans = compute_section_attention_representations_from_positions(
                teacher_hidden_state[batch_idx],
                teacher_cypher_spans,
                token_positions,
            )
            if student_section_spans is None or teacher_section_spans is None:
                continue

            student_rel = _safe_cosine_similarity(student_cypher_spans, student_section_spans, dim=-1)
            teacher_rel = _safe_cosine_similarity(teacher_cypher_spans, teacher_section_spans, dim=-1)
            per_span = (student_rel - teacher_rel).pow(2)
            per_span = torch.nan_to_num(per_span, nan=0.0, posinf=4.0, neginf=0.0).clamp(min=0.0, max=4.0)

            loss_nums[section_idx] = loss_nums[section_idx] + (per_span * weights).sum()
            loss_dens[section_idx] = loss_dens[section_idx] + weights.sum()

    losses = []
    for loss_num, loss_den in zip(loss_nums, loss_dens):
        if loss_den <= 0:
            losses.append(zero)
            continue
        loss = loss_num / loss_den.clamp(min=1e-5)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=4.0, neginf=0.0).clamp(min=0.0, max=4.0)
        losses.append(loss)
    return tuple(losses)


def compute_grounding_losses_for_layer(
    student_hidden_state,
    teacher_hidden_state,
    token_to_span_map,
    span_mask,
    query_mask,
    schema_mask,
    schema_linking_mask,
):
    section_losses = compute_aligned_span_relation_losses_for_sections(
        student_hidden_state,
        teacher_hidden_state,
        token_to_span_map,
        span_mask,
        (query_mask, schema_mask, schema_linking_mask),
    )
    return tuple(
        torch.nan_to_num(loss, nan=0.0, posinf=4.0, neginf=0.0).clamp(min=0.0, max=4.0)
        for loss in section_losses
    )


def compute_overall_relation_loss(
    tokenizer,
    input_ids,
    attention_mask,
    labels,
    student_hidden_states,
    teacher_hidden_states,
    offsets_mapping,
    spans_offsets,
    args,
):
    token_to_span_map, span_mask, schema_linking_offset_mask = prepare_span_token_map(
        attention_mask,
        offsets_mapping,
        spans_offsets,
    )
    if token_to_span_map is None:
        zero = _zero_loss(attention_mask.device)
        return zero, zero, zero

    query_mask, schema_mask = build_prompt_section_masks(input_ids, attention_mask, labels, tokenizer)
    schema_linking_marker_mask = build_response_schema_linking_mask(input_ids, attention_mask, labels, tokenizer)
    schema_linking_mask = schema_linking_marker_mask | schema_linking_offset_mask
    if not query_mask.any() and not schema_mask.any() and not schema_linking_mask.any():
        zero = _zero_loss(attention_mask.device)
        return zero, zero, zero

    query_rel_total = _zero_loss(attention_mask.device)
    schema_rel_total = _zero_loss(attention_mask.device)
    schema_linking_rel_total = _zero_loss(attention_mask.device)
    valid_layers = 0

    for student_idx, teacher_idx in zip(args.student_layer_mapping, args.teacher_layer_mapping):
        student_hidden = (
            student_hidden_states.get(student_idx)
            if isinstance(student_hidden_states, dict)
            else student_hidden_states[student_idx]
        )
        teacher_hidden = teacher_hidden_states[teacher_idx]
        if student_hidden is None:
            continue

        query_rel_loss, schema_rel_loss, schema_linking_rel_loss = compute_grounding_losses_for_layer(
            student_hidden,
            teacher_hidden,
            token_to_span_map,
            span_mask,
            query_mask,
            schema_mask,
            schema_linking_mask,
        )
        query_rel_total += query_rel_loss
        schema_rel_total += schema_rel_loss
        schema_linking_rel_total += schema_linking_rel_loss
        valid_layers += 1

    if valid_layers == 0:
        zero = _zero_loss(attention_mask.device)
        return zero, zero, zero

    return (
        query_rel_total / valid_layers,
        schema_rel_total / valid_layers,
        schema_linking_rel_total / valid_layers,
    )


def _resolve_student_hook_layers(student_layer_mapping, layers):
    num_layers = len(layers)
    hook_layers = {}
    for capture_idx in set(student_layer_mapping):
        if capture_idx == 0:
            continue

        layer_idx = capture_idx - 1 if capture_idx > 0 else num_layers + capture_idx
        if layer_idx < 0 or layer_idx >= num_layers:
            raise IndexError(
                f"student_layer_mapping index {capture_idx} is out of range for {num_layers} transformer layers"
            )
        hook_layers[capture_idx] = layer_idx
    return hook_layers


def finetune(
    args,
    tokenizer: AutoTokenizer,
    model: deepspeed.DeepSpeedEngine,
    optimizer: AdamW,
    lr_scheduler,
    dataset,
    device,
    teacher_model=None,
):
    print_rank("Start Fine-tuning with attention-weighted query/schema grounding losses")

    if args.model_parallel:
        raise NotImplementedError

    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    dp_group = None
    loss_func = nn.CrossEntropyLoss()
    grounding_cfg = get_grounding_loss_config(args)
    use_grounding_loss = teacher_model is not None and any(weight > 0.0 for weight in grounding_cfg.values())

    sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
    train_dataloader = DataLoader(
        dataset["train"],
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset["train"].collate,
    )

    if "pt_train" in dataset:
        pt_sampler = DistributedSampler(dataset["pt_train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
        pt_train_dataloader = DataLoader(
            dataset["pt_train"],
            sampler=pt_sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=dataset["pt_train"].collate,
        )
        pt_train_iter = iter(pt_train_dataloader)

    student_generator = SampleGenerator(args, tokenizer)

    step, global_step = 1, 1
    total_loss, total_distil_loss, total_grounding_loss, total_time = 0.0, 0.0, 0.0, 0.0
    total_span_query_loss = 0.0
    total_span_schema_loss = 0.0
    total_span_schema_linking_loss = 0.0

    adaptive_threshold = args.init_threshold if "adaptive" in args.type else -1.0
    prev_avg_loss = 0.0
    replay_buffer = ReplayBuffer(args)

    student_captured_hidden = {0: None}
    hook_handles = []

    def make_capture_hook(capture_idx):
        def capture_hook_fn(module, inputs, output):
            if module.training:
                student_captured_hidden[capture_idx] = output[0] if isinstance(output, tuple) else output

        return capture_hook_fn

    if use_grounding_loss:
        student_layers = model.base_model.model.model.layers
        for capture_idx, layer_idx in _resolve_student_hook_layers(args.student_layer_mapping, student_layers).items():
            hook_handles.append(student_layers[layer_idx].register_forward_hook(make_capture_hook(capture_idx)))

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        alloc_sum = 0.0
        alloc_count = 0

        model.train()
        for _, (model_batch, no_model_batch, gen_data, _, _) in enumerate(train_dataloader):
            dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, device)
            student_captured_hidden.clear()
            student_captured_hidden[0] = None

            if args.lm_data_dir is not None:
                try:
                    pt_model_batch, pt_no_model_batch, pt_gen_data = next(pt_train_iter)
                except Exception:
                    pt_train_iter = iter(pt_train_dataloader)
                    pt_model_batch, pt_no_model_batch, pt_gen_data = next(pt_train_iter)
                dataset["pt_train"].move_to_device(pt_model_batch, pt_no_model_batch, pt_gen_data, device)

            torch.cuda.synchronize()
            st_time = time.time()

            samp_threshold = adaptive_threshold * (1 - global_step / args.total_iters)
            if "adaptive" in args.type:
                if args.replay_ratio == "constant":
                    samp_threshold = adaptive_threshold * 0.5
                elif args.replay_ratio == "increasing":
                    samp_threshold = adaptive_threshold * global_step / args.total_iters
                else:
                    samp_threshold = adaptive_threshold * (1 - global_step / args.total_iters)

            if args.student_gen:
                rand_value = np.random.uniform(0, 1)
                if "mixed" in args.type and rand_value < args.mixed_alpha:
                    model_batch = student_generator.run_sample(model, gen_data)
                    no_model_batch = build_generated_no_model_batch(model_batch.pop("no_model_batch"))
                    replay_buffer.move_to_memory(model_batch, no_model_batch)
                    model_batch, no_model_batch = replay_buffer.sample()
                    model_batch, no_model_batch = replay_buffer.move_to_device(
                        model_batch,
                        no_model_batch,
                        device,
                    )
                elif "adaptive" in args.type and (
                    rand_value < samp_threshold
                    or (rand_value < adaptive_threshold and len(replay_buffer) < args.capacity)
                ):
                    model_batch = student_generator.run_sample(model, gen_data)
                    no_model_batch = build_generated_no_model_batch(model_batch.pop("no_model_batch"))
                    if args.model_type in ["opt"]:
                        model_batch.pop("position_ids", None)
                    replay_buffer.move_to_memory(model_batch, no_model_batch)
                elif "adaptive" in args.type and rand_value < adaptive_threshold:
                    model_batch, no_model_batch = replay_buffer.sample()
                    model_batch, no_model_batch = replay_buffer.move_to_device(model_batch, no_model_batch, device)
                model.train()

            outputs = model(**model_batch, use_cache=False)
            logits = outputs.logits
            lm_loss = loss_func(logits.float().reshape(-1, logits.shape[-1]), no_model_batch["label"].view(-1))

            weighted_grounding_loss = logits.new_tensor(0.0)
            weighted_span_query_loss = logits.new_tensor(0.0)
            weighted_span_schema_loss = logits.new_tensor(0.0)
            weighted_span_schema_linking_loss = logits.new_tensor(0.0)
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_model.eval()
                    teacher_outputs = teacher_model(
                        **model_batch,
                        output_hidden_states=use_grounding_loss,
                        use_cache=False,
                    )
                    teacher_logits = teacher_outputs.logits

                distil_loss = get_distil_loss(args, teacher_logits, no_model_batch, logits)
                distil_loss = torch.nan_to_num(distil_loss, nan=0.0, posinf=100.0, neginf=0.0)
                if use_grounding_loss and "offset_mapping" in no_model_batch and "span_offsets" in no_model_batch:
                    span_query_loss, span_schema_loss, span_schema_linking_loss = compute_overall_relation_loss(
                        tokenizer,
                        model_batch["input_ids"],
                        model_batch["attention_mask"],
                        no_model_batch["label"],
                        student_captured_hidden,
                        teacher_outputs.hidden_states,
                        no_model_batch["offset_mapping"],
                        no_model_batch["span_offsets"],
                        args,
                    )
                else:
                    span_query_loss = logits.new_tensor(0.0)
                    span_schema_loss = logits.new_tensor(0.0)
                    span_schema_linking_loss = logits.new_tensor(0.0)

                weighted_span_query_loss = torch.nan_to_num(
                    grounding_cfg["w_span_query"] * span_query_loss,
                    nan=0.0, posinf=1e4, neginf=0.0,
                ).clamp(min=0.0, max=1e4)
                weighted_span_schema_loss = torch.nan_to_num(
                    grounding_cfg["w_span_schema"] * span_schema_loss,
                    nan=0.0, posinf=1e4, neginf=0.0,
                ).clamp(min=0.0, max=1e4)
                weighted_span_schema_linking_loss = torch.nan_to_num(
                    grounding_cfg["w_span_schema_linking"] * span_schema_linking_loss,
                    nan=0.0, posinf=1e4, neginf=0.0,
                ).clamp(min=0.0, max=1e4)
                weighted_grounding_loss = (
                    weighted_span_query_loss + weighted_span_schema_loss + weighted_span_schema_linking_loss
                ).clamp(min=0.0, max=1e4)

                distil_loss = distil_loss + weighted_grounding_loss
                loss = (1 - args.kd_ratio) * lm_loss + args.kd_ratio * distil_loss
            else:
                distil_loss = logits.new_tensor(0.0)
                loss = lm_loss

            if args.lm_data_dir is not None:
                assert args.lm_coef is not None
                loss = loss + args.lm_coef * pt_loss(args, model, pt_model_batch, pt_no_model_batch)

            loss = torch.nan_to_num(loss, nan=0.0, posinf=100.0, neginf=0.0)

            model.backward(loss)
            model.step()

            global_distil_loss = 0.0
            global_grounding_loss = 0.0
            global_span_query_loss = 0.0
            global_span_schema_loss = 0.0
            global_span_schema_linking_loss = 0.0
            if teacher_model is not None:
                reduced_metrics = torch.stack(
                    [
                        loss.detach().float(),
                        distil_loss.detach().float(),
                        weighted_grounding_loss.detach().float(),
                        weighted_span_query_loss.detach().float(),
                        weighted_span_schema_loss.detach().float(),
                        weighted_span_schema_linking_loss.detach().float(),
                    ]
                )
                dist.all_reduce(reduced_metrics, dist.ReduceOp.SUM, group=dp_group)
                (
                    global_loss,
                    global_distil_loss,
                    global_grounding_loss,
                    global_span_query_loss,
                    global_span_schema_loss,
                    global_span_schema_linking_loss,
                ) = (reduced_metrics / dp_world_size).tolist()

                total_distil_loss += global_distil_loss
                total_grounding_loss += global_grounding_loss
                total_span_query_loss += global_span_query_loss
                total_span_schema_loss += global_span_schema_loss
                total_span_schema_linking_loss += global_span_schema_linking_loss
            else:
                reduced_loss = loss.detach().float()
                dist.all_reduce(reduced_loss, dist.ReduceOp.SUM, group=dp_group)
                global_loss = reduced_loss.item() / dp_world_size

            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time
            total_loss += global_loss
            total_time += elapsed_time

            def get_log(
                log_loss,
                log_distil_loss,
                log_ground,
                log_span_query,
                log_span_schema,
                log_span_schema_linking,
                log_time,
            ):
                return (
                    "train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | "
                    "loss: {:.4f} | ds_loss: {:.4f} | ground: {:.4f} | "
                    "span_q: {:.4f} | span_s: {:.4f} | span_sl: {:.4f} | lr: {:.4e} | "
                    "scale: {:10.4f} | micro time: {:.3f} | step time: {:.3f}"
                ).format(
                    epoch,
                    step,
                    args.total_iters * args.gradient_accumulation_steps,
                    global_step,
                    args.total_iters,
                    log_loss,
                    log_distil_loss,
                    log_ground,
                    log_span_query,
                    log_span_schema,
                    log_span_schema_linking,
                    lr_scheduler.get_last_lr()[0],
                    optimizer.cur_scale if hasattr(optimizer, "cur_scale") else 0,
                    elapsed_time,
                    log_time,
                )

            if args.mid_log_num > 0:
                mid_log_step = max(1, args.gradient_accumulation_steps // args.mid_log_num)
                if step % mid_log_step == 0:
                    print_rank(
                        get_log(
                            global_loss,
                            global_distil_loss,
                            global_grounding_loss,
                            global_span_query_loss,
                            global_span_schema_loss,
                            global_span_schema_linking_loss,
                            0.0,
                        )
                    )

            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                denom = args.log_interval * args.gradient_accumulation_steps
                log_str = get_log(
                    total_loss / denom,
                    total_distil_loss / denom,
                    total_grounding_loss / denom,
                    total_span_query_loss / denom,
                    total_span_schema_loss / denom,
                    total_span_schema_linking_loss / denom,
                    total_time / args.log_interval,
                )
                print_rank("*" * 100)
                print_rank(log_str)
                print_rank(args.save)
                print_rank("*" * 100)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                total_loss, total_distil_loss, total_grounding_loss, total_time = 0.0, 0.0, 0.0, 0.0
                total_span_query_loss = 0.0
                total_span_schema_loss = 0.0
                total_span_schema_linking_loss = 0.0

                allocated = torch.cuda.memory_allocated() / 1e9
                peak_alloc = torch.cuda.max_memory_allocated() / 1e9
                alloc_sum += allocated
                alloc_count += 1
                avg_alloc = alloc_sum / alloc_count
                print_rank("train | avg_alloc {:.4f} GB | peak_alloc {:.4f} GB".format(avg_alloc, peak_alloc))

            if args.save and args.save_interval and global_step % args.save_interval == 0 and step % args.gradient_accumulation_steps == 0:
                save_dir_path = os.path.join(args.save, str(global_step))
                if dist.get_rank() == 0:
                    os.makedirs(save_dir_path, exist_ok=True)
                    print_rank(f"Model save to {save_dir_path}")
                    tokenizer.save_pretrained(save_dir_path)
                    model.module.save_pretrained(save_dir_path, safe_serialization=False)
                dist.barrier()

            if args.eval_interval and global_step % args.eval_interval == 0 and step % args.gradient_accumulation_steps == 0:
                curr_avg_loss = evaluate(args, tokenizer, model, dataset["dev"], "dev", epoch, device, adaptive_threshold)
                if "adaptive" in args.type and curr_avg_loss >= prev_avg_loss + args.loss_eps:
                    adaptive_threshold = min(adaptive_threshold + 0.1, 1.0)
                    prev_avg_loss = curr_avg_loss
                # evaluate(args, tokenizer, model, dataset["test"], "test", epoch, device)
                model.train()

            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1
            if global_step > args.total_iters:
                break

    for handle in hook_handles:
        handle.remove()

    return model


def main():
    torch.backends.cudnn.enabled = False

    args = get_args()
    initialize(args)

    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            import json

            json.dump(vars(args), f)

    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "=" * 30 + f" EXP at {cur_time} " + "=" * 30, os.path.join(args.save, "log.txt"))

    with open(args.deepspeed_config, "r") as f:
        import json

        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000

    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0

    args.fp32 = not ds_config["fp16"]["enabled"]
    args.bf16 = "bf16" in ds_config and ds_config["bf16"]["enabled"]
    args.deepspeed_config = None

    tokenizer = get_tokenizer(args)
    print(type(tokenizer))

    patch_response_span_extractor_for_new_data()
    dataset = prepare_dataset_new_data_safe(args, tokenizer)
    dp_world_size = dist.get_world_size()

    if args.do_train:
        args.train_iters_per_epoch = int(
            len(dataset["train"]) / (args.batch_size * dp_world_size * args.gradient_accumulation_steps)
        )
        print_rank("Train iters per epoch", args.train_iters_per_epoch)
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.epochs
        if args.epochs is None:
            args.epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)
        print_rank("total_iters", args.total_iters)

        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch

    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, ds_config, device, set_optim=args.do_train)

    if args.teacher_model_type is None:
        args.teacher_model_type = args.model_type
    teacher_model = get_teacher_model(args, device) if args.teacher_model_path is not None else None

    if args.do_train:
        model = finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, device, teacher_model=teacher_model)

    if args.do_eval:
        pass
        # evaluate(args, tokenizer, model, dataset["test"], "test", 0, device)


if __name__ == "__main__":
    main()
