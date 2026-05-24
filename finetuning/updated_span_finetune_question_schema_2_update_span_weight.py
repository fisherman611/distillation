import math
import os
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

from arguments import get_args
from data_utils.lm_datasets_new import extract_event_span_offsets, extract_text2cypher_span_offsets
from distillm import ReplayBuffer, SampleGenerator
try:
    from .span_finetune import (
        evaluate,
        get_distil_loss,
        get_teacher_model,
        prepare_dataset,
        pt_loss,
        setup_model_and_optimizer,
    )
except ImportError:
    from span_finetune import (
        evaluate,
        get_distil_loss,
        get_teacher_model,
        prepare_dataset,
        pt_loss,
        setup_model_and_optimizer,
    )
from utils import get_tokenizer, initialize, print_args, print_rank, save_rank


torch.set_num_threads(4)
_TOKENIZED_MARKER_CACHE = {}
MAX_REL_LOSS = 4.0
MAX_GROUNDING_LOSS = 1e4
NEG_INF = -1e4
EPS = 1e-5

QUESTION_MARKER_TEXT = "QUESTION:\n"
SCHEMA_MARKER_TEXT = "SCHEMA:\n"
SCHEMA_END_MARKER_TEXT = (
    "\n\nGenerate a Cypher query that answers the question using only the provided schema.\n"
    "Return only the JSON object in the required format."
)


def get_grounding_loss_config(args):
    w_rel = float(getattr(args, "w_rel_loss", 1.0))
    if (not math.isfinite(w_rel)) or w_rel < 0.0:
        w_rel = 1.0
    return {"w_rel": min(w_rel, 1e4)}


def build_generated_no_model_batch(tokenizer, args, model_batch, labels):
    loss_mask = (labels != -100).float()
    device = labels.device
    input_ids = model_batch["input_ids"].detach().cpu()
    attention_mask = model_batch["attention_mask"].detach().cpu().bool()
    labels_cpu = labels.detach().cpu()

    offset_mappings = []
    span_offsets = []
    for idx in range(input_ids.size(0)):
        valid_mask = attention_mask[idx]
        response_mask = labels_cpu[idx] != -100
        prompt_mask = valid_mask & ~response_mask

        prompt_ids = input_ids[idx][prompt_mask].tolist()
        response_ids = labels_cpu[idx][response_mask].tolist()
        prompt_text = tokenizer.decode(
            prompt_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        response_text = tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        full_text = prompt_text + response_text

        encoded = tokenizer(
            full_text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
            add_special_tokens=False,
            return_tensors="pt",
        )
        offset_mappings.append(encoded["offset_mapping"])

        sample_spans = extract_text2cypher_span_offsets(full_text, response_text)
        if not sample_spans:
            sample_spans = extract_event_span_offsets(full_text, response_text)
        span_offsets.append(sample_spans)

    return {
        "label": labels,
        "loss_mask": loss_mask,
        "offset_mapping": torch.cat(offset_mappings, dim=0).to(device),
        "span_offsets": span_offsets,
    }


def _zero_scalar_like(tensor):
    return tensor.new_tensor(0.0)


def _sanitize_loss(value, cap):
    return torch.nan_to_num(value, nan=0.0, posinf=cap, neginf=0.0).clamp(min=0.0, max=cap)


def build_prompt_token_mask(attention_mask, labels):
    valid_token_mask = attention_mask.bool()
    prompt_mask = (labels == -100) & valid_token_mask
    no_prompt = (~prompt_mask.any(dim=-1)) & valid_token_mask.any(dim=-1)
    if no_prompt.any():
        prompt_mask[no_prompt] = valid_token_mask[no_prompt]
    return prompt_mask


def _tokenize_marker(tokenizer, marker_text):
    return tokenizer.encode(marker_text, add_special_tokens=False)


def _find_subsequence(sequence, pattern, start=0):
    if not pattern:
        return -1, 0

    max_start = len(sequence) - len(pattern)
    if start > max_start:
        return -1, 0

    for idx in range(max(0, start), max_start + 1):
        if sequence[idx : idx + len(pattern)] == pattern:
            return idx, len(pattern)
    return -1, 0


def _get_tokenized_markers(tokenizer):
    cache_key = id(tokenizer)
    markers = _TOKENIZED_MARKER_CACHE.get(cache_key)
    if markers is None:
        markers = {
            "question": _tokenize_marker(tokenizer, QUESTION_MARKER_TEXT),
            "schema": _tokenize_marker(tokenizer, SCHEMA_MARKER_TEXT),
            "schema_end": _tokenize_marker(tokenizer, SCHEMA_END_MARKER_TEXT),
        }
        _TOKENIZED_MARKER_CACHE[cache_key] = markers
    return markers


def build_question_schema_context_mask(tokenizer, input_ids, attention_mask, labels):
    prompt_mask = build_prompt_token_mask(attention_mask, labels)
    source_mask = torch.zeros_like(prompt_mask)
    markers = _get_tokenized_markers(tokenizer)

    for batch_idx in range(input_ids.size(0)):
        prompt_indices = torch.nonzero(prompt_mask[batch_idx], as_tuple=False).flatten()
        if prompt_indices.numel() == 0:
            continue

        token_ids = input_ids[batch_idx, prompt_indices].detach().cpu().tolist()
        question_pos, question_len = _find_subsequence(token_ids, markers["question"])
        if question_pos < 0:
            continue

        schema_pos, schema_len = _find_subsequence(token_ids, markers["schema"], start=question_pos + question_len)
        if schema_pos < 0:
            continue

        question_start = question_pos + question_len
        question_end = schema_pos

        schema_start = schema_pos + schema_len
        schema_end, _ = _find_subsequence(token_ids, markers["schema_end"], start=schema_start)
        if schema_end < 0:
            schema_end = len(token_ids)

        if question_start < question_end:
            source_mask[batch_idx, prompt_indices[question_start:question_end]] = True
        if schema_start < schema_end:
            source_mask[batch_idx, prompt_indices[schema_start:schema_end]] = True

    return source_mask & attention_mask.bool()


def build_token_to_span_map(attention_mask, offsets_mapping, spans_offsets):
    device = attention_mask.device
    batch_size, seq_len = attention_mask.shape

    max_spans = max((len(sample_spans) for sample_spans in spans_offsets), default=0)
    if max_spans == 0:
        return None, None

    span_starts = torch.zeros(batch_size, max_spans, dtype=torch.long, device=device)
    span_ends = torch.zeros(batch_size, max_spans, dtype=torch.long, device=device)
    span_mask = torch.zeros(batch_size, max_spans, dtype=torch.bool, device=device)

    for batch_idx, sample_spans in enumerate(spans_offsets):
        if not sample_spans:
            continue
        spans_tensor = torch.tensor(sample_spans, dtype=torch.long, device=device)
        span_starts[batch_idx, : len(sample_spans)] = spans_tensor[:, 0]
        span_ends[batch_idx, : len(sample_spans)] = spans_tensor[:, 1]
        span_mask[batch_idx, : len(sample_spans)] = True

    current_offsets = offsets_mapping[:, :seq_len, :] if offsets_mapping.shape[1] != seq_len else offsets_mapping
    token_start = current_offsets[..., 0].unsqueeze(-1).to(device)
    token_end = current_offsets[..., 1].unsqueeze(-1).to(device)

    token_in_span = (token_start + 1 >= span_starts.unsqueeze(1)) & (token_end <= span_ends.unsqueeze(1))
    token_in_span = token_in_span & attention_mask.unsqueeze(-1).bool() & span_mask.unsqueeze(1)

    if not token_in_span.any():
        return None, None

    return token_in_span, span_mask


def compute_token_weights(hidden_state, attention_mask):
    hidden_state = hidden_state.float()
    attention_mask = attention_mask.bool()

    #  H_hat_{t,l} = H_{t,l} / sigma(H_{t,l})
    std = hidden_state.std(dim=-1, keepdim=True, unbiased=False).clamp(min=EPS)
    standardized_hidden = hidden_state / std
    # pairwise score alpha_{s->t,l} with padding/diagonal masking.
    scores = torch.matmul(standardized_hidden, standardized_hidden.transpose(1, 2)) / math.sqrt(hidden_state.size(-1))

    valid_pair_mask = attention_mask.unsqueeze(1) & attention_mask.unsqueeze(2)
    scores = scores.masked_fill(~valid_pair_mask, NEG_INF)

    diag_mask = torch.eye(scores.size(-1), device=scores.device, dtype=torch.bool).unsqueeze(0)
    scores = scores.masked_fill(diag_mask, NEG_INF)

    attention_probs = torch.softmax(scores, dim=-1)
    attention_probs = torch.nan_to_num(attention_probs, nan=0.0, posinf=0.0, neginf=0.0)
    attention_probs = attention_probs * valid_pair_mask.float()
    attention_probs = attention_probs / attention_probs.sum(dim=-1, keepdim=True).clamp(min=EPS)

    # w_{t,l} = (1/N) * sum_s sigma_{s->t,l}, with N = number of valid source tokens.
    valid_source_count = attention_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
    token_weights = attention_probs.sum(dim=1) / valid_source_count
    token_weights = token_weights * attention_mask.float()
    return token_weights


def compute_weighted_span_representations(hidden_state, token_to_span_map, span_mask, token_weights):
    # U_{k,l} = sum_{t in S_k}(w_{t,l} * H_{t,l}) / sum_{t in S_k}(w_{t,l})
    token_to_span_map = token_to_span_map.float()
    token_weights = token_weights.float().unsqueeze(-1)
    weighted_token_to_span = token_to_span_map * token_weights
    span_sums = torch.einsum("bld,bls->bsd", hidden_state.float(), weighted_token_to_span)
    span_weight_sums = weighted_token_to_span.sum(dim=1).unsqueeze(-1).clamp(min=EPS)
    span_repr = span_sums / span_weight_sums
    span_repr = span_repr * span_mask.unsqueeze(-1).float()
    return span_repr


def compute_span_context_vectors(
    hidden_state,
    span_repr,
    span_mask,
    source_mask,
    shared_attn_weights=None,
):
    hidden_state = hidden_state.float()
    span_repr = span_repr.float()

    if shared_attn_weights is None:
        scores = torch.matmul(span_repr, hidden_state.transpose(1, 2))
        scores = scores / math.sqrt(hidden_state.size(-1))
        # Use a finite large negative number to keep softmax numerically stable.
        scores = scores.masked_fill(~source_mask.unsqueeze(1), NEG_INF)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
        attn_weights = attn_weights * source_mask.unsqueeze(1).float()
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp(min=EPS)
    else:
        attn_weights = shared_attn_weights.float()

    attn_weights = attn_weights * span_mask.unsqueeze(-1).float()

    context_repr = torch.matmul(attn_weights, hidden_state)
    context_repr = context_repr * span_mask.unsqueeze(-1).float()

    return context_repr, attn_weights


def _safe_cosine_similarity(x, y, dim=-1, eps=1e-6):
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    y = torch.nan_to_num(y.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    x = F.normalize(x, p=2, dim=dim, eps=eps)
    y = F.normalize(y, p=2, dim=dim, eps=eps)
    return (x * y).sum(dim=dim).clamp(min=-1.0, max=1.0)


def compute_layer_span_context_relation_loss(
    student_span,
    student_context,
    teacher_span,
    teacher_context,
    span_mask,
    span_weights,
    use_span_weight=True,
):
    student_rel = _safe_cosine_similarity(student_span, student_context, dim=-1)
    teacher_rel = _safe_cosine_similarity(teacher_span, teacher_context, dim=-1)
    rel_diff = student_rel - teacher_rel
    per_span = rel_diff.pow(2)
    per_span = _sanitize_loss(per_span, MAX_REL_LOSS)
    if use_span_weight:
        # Span weight from teacher token importance, normalized across spans in each sample.
        weights = span_weights * span_mask.float()
    else:
        weights = span_mask.float()
    loss = (per_span * weights).sum() / weights.sum().clamp(min=EPS)
    return _sanitize_loss(loss, MAX_REL_LOSS)


def compute_grounding_losses_for_layer(
    student_hidden_state,
    teacher_hidden_state,
    token_to_span_map,
    span_mask,
    source_mask,
    attention_mask,
    use_teacher_shared_attn=True,
    use_span_weight=True,
):
    valid_sample_mask = span_mask.any(dim=-1) & source_mask.any(dim=-1)
    if not valid_sample_mask.any():
        return _zero_scalar_like(student_hidden_state)

    student_hidden_state = student_hidden_state[valid_sample_mask]
    teacher_hidden_state = teacher_hidden_state[valid_sample_mask]
    token_to_span_map = token_to_span_map[valid_sample_mask]
    span_mask = span_mask[valid_sample_mask]
    source_mask = source_mask[valid_sample_mask]
    attention_mask = attention_mask[valid_sample_mask]

    # Token importance is defined over valid sequence tokens (not restricted to source context mask).
    student_token_weights = compute_token_weights(student_hidden_state, attention_mask)
    teacher_token_weights = compute_token_weights(teacher_hidden_state, attention_mask)
    raw_span_weights = (token_to_span_map.float() * teacher_token_weights.unsqueeze(-1)).sum(dim=1)
    raw_span_weights = raw_span_weights * span_mask.float()
    span_weight_denom = raw_span_weights.sum(dim=-1, keepdim=True).clamp(min=EPS)
    span_weights = raw_span_weights / span_weight_denom
    student_span = compute_weighted_span_representations(
        student_hidden_state, token_to_span_map, span_mask, student_token_weights
    )
    teacher_span = compute_weighted_span_representations(
        teacher_hidden_state, token_to_span_map, span_mask, teacher_token_weights
    )

    teacher_context, teacher_attn_weights = compute_span_context_vectors(
        teacher_hidden_state,
        teacher_span,
        span_mask,
        source_mask,
    )
    student_context, _ = compute_span_context_vectors(
        student_hidden_state,
        student_span,
        span_mask,
        source_mask,
        shared_attn_weights=teacher_attn_weights if use_teacher_shared_attn else None,
    )
    rel_loss = compute_layer_span_context_relation_loss(
        student_span,
        student_context,
        teacher_span,
        teacher_context,
        span_mask,
        span_weights,
        use_span_weight=use_span_weight,
    )

    return _sanitize_loss(rel_loss, MAX_REL_LOSS)


def _compute_sampling_threshold(args, adaptive_threshold, global_step):
    if "adaptive" not in args.type:
        return adaptive_threshold * (1 - global_step / args.total_iters)
    if args.replay_ratio == "constant":
        return adaptive_threshold * 0.5
    if args.replay_ratio == "increasing":
        return adaptive_threshold * global_step / args.total_iters
    return adaptive_threshold * (1 - global_step / args.total_iters)


def _should_run_interval(global_step, step, interval, grad_accum_steps):
    if not interval:
        return False
    return global_step % interval == 0 and step % grad_accum_steps == 0


def _reduce_metric(value, world_size):
    reduced_value = value.detach().clone()
    dist.all_reduce(reduced_value, dist.ReduceOp.SUM)
    return reduced_value.item() / world_size


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _format_train_log(
    epoch,
    step,
    global_step,
    args,
    loss,
    distil_loss,
    grounding_loss,
    rel_loss,
    lr_scheduler,
    optimizer,
    elapsed_time,
    step_time,
):
    return (
        "train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | "
        "loss: {:.4f} | ds_loss: {:.4f} | ground: {:.4f} | "
        "rel: {:.4f} | lr: {:.4e} | "
        "scale: {:10.4f} | micro time: {:.3f} | step time: {:.3f}"
    ).format(
        epoch,
        step,
        args.total_iters * args.gradient_accumulation_steps,
        global_step,
        args.total_iters,
        loss,
        distil_loss,
        grounding_loss,
        rel_loss,
        lr_scheduler.get_last_lr()[0],
        optimizer.cur_scale if hasattr(optimizer, "cur_scale") else 0,
        elapsed_time,
        step_time,
    )


def compute_multi_layer_span_context_relation_loss(
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
    # Backward compatible flag resolution:
    # prefer `use_span_weight`, fallback to legacy `use_span_length_weight`.
    use_span_weight = getattr(args, "use_span_weight", getattr(args, "use_span_length_weight", True))
    if offsets_mapping is None or spans_offsets is None:
        return _zero_scalar_like(attention_mask)
    if not isinstance(offsets_mapping, torch.Tensor):
        return _zero_scalar_like(attention_mask)
    if not isinstance(spans_offsets, (list, tuple)) or len(spans_offsets) != attention_mask.size(0):
        return _zero_scalar_like(attention_mask)

    token_to_span_map, span_mask = build_token_to_span_map(attention_mask, offsets_mapping, spans_offsets)
    if token_to_span_map is None:
        return _zero_scalar_like(attention_mask)

    source_mask = build_question_schema_context_mask(tokenizer, input_ids, attention_mask, labels)
    if not source_mask.any():
        return _zero_scalar_like(attention_mask)

    rel_total = attention_mask.new_tensor(0.0)
    valid_layers = 0

    for student_idx, teacher_idx in zip(args.student_layer_mapping, args.teacher_layer_mapping):
        student_hidden = student_hidden_states[student_idx]
        teacher_hidden = teacher_hidden_states[teacher_idx]
        if student_hidden is None:
            continue

        rel_loss = compute_grounding_losses_for_layer(
            student_hidden,
            teacher_hidden,
            token_to_span_map,
            span_mask,
            source_mask,
            attention_mask,
            use_teacher_shared_attn=getattr(args, "use_teacher_shared_attn", True),
            use_span_weight=use_span_weight,
        )
        rel_total += rel_loss
        valid_layers += 1

    if valid_layers == 0:
        return _zero_scalar_like(attention_mask)

    return rel_total / valid_layers


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
    print_rank("Start Fine-tuning with updated relation grounding loss")

    if args.model_parallel:
        raise NotImplementedError

    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    loss_func = nn.CrossEntropyLoss()
    grounding_cfg = get_grounding_loss_config(args)

    sampler = DistributedSampler(
        dataset["train"],
        shuffle=True,
        drop_last=True,
        rank=dp_rank,
        num_replicas=dp_world_size,
    )
    train_dataloader = DataLoader(
        dataset["train"],
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset["train"].collate,
    )

    if "pt_train" in dataset:
        pt_sampler = DistributedSampler(
            dataset["pt_train"],
            shuffle=True,
            drop_last=True,
            rank=dp_rank,
            num_replicas=dp_world_size,
        )
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
    total_rel_loss = 0.0

    adaptive_threshold = args.init_threshold if "adaptive" in args.type else -1.0
    prev_avg_loss = 0.0
    replay_buffer = ReplayBuffer(args)

    student_captured_hidden = []
    hook_handles = []

    def capture_hook_fn(layer_module, _, output):
        if layer_module.training:
            student_captured_hidden.append(output[0] if isinstance(output, tuple) else output)

    for layer in model.base_model.model.model.layers:
        hook_handles.append(layer.register_forward_hook(capture_hook_fn))

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)

        model.train()
        for _, (model_batch, no_model_batch, gen_data, _, _) in enumerate(train_dataloader):
            dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, device)
            student_captured_hidden.clear()
            student_captured_hidden.append(None)

            if args.lm_data_dir is not None:
                try:
                    pt_model_batch, pt_no_model_batch, pt_gen_data = next(pt_train_iter)
                except Exception:
                    pt_train_iter = iter(pt_train_dataloader)
                    pt_model_batch, pt_no_model_batch, pt_gen_data = next(pt_train_iter)
                dataset["pt_train"].move_to_device(pt_model_batch, pt_no_model_batch, pt_gen_data, device)

            _sync_cuda()
            st_time = time.time()

            samp_threshold = _compute_sampling_threshold(args, adaptive_threshold, global_step)

            if args.student_gen:
                rand_value = np.random.uniform(0, 1)
                if "mixed" in args.type and rand_value < args.mixed_alpha:
                    model_batch = student_generator.run_sample(model, gen_data)
                    generated_labels = model_batch.pop("no_model_batch")
                    no_model_batch = build_generated_no_model_batch(tokenizer, args, model_batch, generated_labels)
                    replay_buffer.move_to_memory(model_batch, no_model_batch, gen_data)
                    model_batch, no_model_batch, gen_data = replay_buffer.sample()
                    model_batch, no_model_batch, gen_data = replay_buffer.move_to_device(
                        model_batch, no_model_batch, gen_data, device
                    )
                elif "adaptive" in args.type and (
                    rand_value < samp_threshold
                    or (rand_value < adaptive_threshold and len(replay_buffer) < args.capacity)
                ):
                    model_batch = student_generator.run_sample(model, gen_data)
                    generated_labels = model_batch.pop("no_model_batch")
                    no_model_batch = build_generated_no_model_batch(tokenizer, args, model_batch, generated_labels)
                    if args.model_type in ["opt"]:
                        model_batch.pop("position_ids")
                    replay_buffer.move_to_memory(model_batch, no_model_batch, gen_data)
                elif "adaptive" in args.type and rand_value < adaptive_threshold:
                    model_batch, no_model_batch, gen_data = replay_buffer.sample()
                    model_batch, no_model_batch, gen_data = replay_buffer.move_to_device(model_batch, no_model_batch, gen_data, device)
                model.train()

            outputs = model(**model_batch, use_cache=False)
            logits = outputs.logits
            lm_loss = loss_func(logits.float().reshape(-1, logits.shape[-1]), no_model_batch["label"].view(-1))

            weighted_grounding_loss = logits.new_tensor(0.0)
            weighted_rel_loss = logits.new_tensor(0.0)
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_model.eval()
                    teacher_outputs = teacher_model(**model_batch, output_hidden_states=True, use_cache=False)
                    teacher_logits = teacher_outputs.logits

                distil_loss = get_distil_loss(args, teacher_logits, no_model_batch, logits)
                distil_loss = torch.nan_to_num(distil_loss, nan=0.0, posinf=100.0, neginf=0.0)
                rel_loss = compute_multi_layer_span_context_relation_loss(
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

                weighted_rel_loss = grounding_cfg["w_rel"] * rel_loss
                weighted_grounding_loss = torch.nan_to_num(
                    weighted_rel_loss,
                    nan=0.0,
                    posinf=MAX_GROUNDING_LOSS,
                    neginf=0.0,
                )
                weighted_grounding_loss = weighted_grounding_loss.clamp(min=0.0, max=MAX_GROUNDING_LOSS)

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

            global_loss = _reduce_metric(loss, dp_world_size)

            global_distil_loss = 0.0
            global_grounding_loss = 0.0
            global_rel_loss = 0.0
            if teacher_model is not None:
                global_distil_loss = _reduce_metric(distil_loss, dp_world_size)
                global_grounding_loss = _reduce_metric(weighted_grounding_loss, dp_world_size)
                global_rel_loss = _reduce_metric(weighted_rel_loss, dp_world_size)

                total_distil_loss += global_distil_loss
                total_grounding_loss += global_grounding_loss
                total_rel_loss += global_rel_loss

            _sync_cuda()
            elapsed_time = time.time() - st_time
            total_loss += global_loss
            total_time += elapsed_time

            if _should_run_interval(global_step, step, args.log_interval, args.gradient_accumulation_steps):
                denom = args.log_interval * args.gradient_accumulation_steps
                avg_distil = total_distil_loss / denom
                avg_rel = total_rel_loss / denom
                log_str = _format_train_log(
                    epoch,
                    step,
                    global_step,
                    args,
                    total_loss / denom,
                    avg_distil,
                    total_grounding_loss / denom,
                    avg_rel,
                    lr_scheduler,
                    optimizer,
                    elapsed_time,
                    total_time / args.log_interval,
                )
                print_rank(log_str)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                total_loss, total_distil_loss, total_grounding_loss, total_time = 0.0, 0.0, 0.0, 0.0
                total_rel_loss = 0.0

            if args.save and _should_run_interval(global_step, step, args.save_interval, args.gradient_accumulation_steps):
                save_dir_path = os.path.join(args.save, str(global_step))
                if dist.get_rank() == 0:
                    os.makedirs(save_dir_path, exist_ok=True)
                    print_rank(f"Model save to {save_dir_path}")
                    tokenizer.save_pretrained(save_dir_path)
                    model.module.save_pretrained(save_dir_path, safe_serialization=False)
                dist.barrier()

            if _should_run_interval(global_step, step, args.eval_interval, args.gradient_accumulation_steps):
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
    dataset = prepare_dataset(args, tokenizer)
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
