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

from arguments import get_args
from distillm import ReplayBuffer, SampleGenerator
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


def get_grounding_loss_config(args):
    w_rel = float(getattr(args, "w_rel_loss", 0.0))
    if w_rel == 0.0:
        w_rel = float(getattr(args, "w_span_loss", 1.0))
    if (not math.isfinite(w_rel)) or w_rel < 0.0:
        w_rel = 1.0

    w_span_query = getattr(args, "w_span_query_loss", None)
    w_span_schema = getattr(args, "w_span_schema_loss", None)
    if w_span_query is None and w_span_schema is None:
        w_span_query = 0.5*w_rel
        w_span_schema = 0.5*w_rel
    elif w_span_query is None:
        w_span_query = 0.0
    elif w_span_schema is None:
        w_span_schema = 0.0

    w_span_query = float(w_span_query)
    w_span_schema = float(w_span_schema)
    if (not math.isfinite(w_span_query)) or w_span_query < 0.0:
        w_span_query = 0.0
    if (not math.isfinite(w_span_schema)) or w_span_schema < 0.0:
        w_span_schema = 0.0

    return {
        "w_span_query": min(w_span_query, 1e4),
        "w_span_schema": min(w_span_schema, 1e4),
    }


def build_prompt_token_mask(attention_mask, labels):
    valid_token_mask = attention_mask.bool()
    prompt_mask = (labels == -100) & valid_token_mask
    no_prompt = (~prompt_mask.any(dim=-1)) & valid_token_mask.any(dim=-1)
    if no_prompt.any():
        prompt_mask[no_prompt] = valid_token_mask[no_prompt]
    return prompt_mask


def _tokenize_marker(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def _find_subsequence(sequence, pattern, start=0):
    if not pattern:
        return -1, 0

    for idx in range(start, len(sequence)):
        end = idx + len(pattern)
        if end <= len(sequence) and sequence[idx:end] == pattern:
            return idx, len(pattern)
    return -1, 0


def build_prompt_section_masks(input_ids, attention_mask, labels, tokenizer):
    prompt_mask = build_prompt_token_mask(attention_mask, labels)
    query_mask = torch.zeros_like(prompt_mask)
    schema_mask = torch.zeros_like(prompt_mask)

    # Keep marker extraction strict and deterministic for the processed prompt format:
    # QUESTION:\n... \n\nSCHEMA:\n... \n\nGenerate a Cypher query ...
    question_marker = _tokenize_marker(
        tokenizer,
        "QUESTION:\n",
    )
    schema_marker = _tokenize_marker(
        tokenizer,
        "SCHEMA:\n",
    )
    schema_end_marker = _tokenize_marker(
        tokenizer,
        (
            "\n\nGenerate a Cypher query that answers the question using only the provided schema.\n"
            "Return only the JSON object in the required format."
        ),
    )

    for batch_idx in range(input_ids.size(0)):
        prompt_indices = torch.nonzero(prompt_mask[batch_idx], as_tuple=False).flatten()
        if prompt_indices.numel() == 0:
            continue

        token_ids = input_ids[batch_idx, prompt_indices].detach().cpu().tolist()
        question_pos, question_len = _find_subsequence(token_ids, question_marker)
        schema_pos, schema_len = _find_subsequence(token_ids, schema_marker, start=max(question_pos + question_len, 0))
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


def prepare_span_token_map(attention_mask, offsets_mapping, spans_offsets):
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


def _safe_cosine_similarity(x, y, dim=-1, eps=1e-6):
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    y = torch.nan_to_num(y.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    x = F.normalize(x, p=2, dim=dim, eps=eps)
    y = F.normalize(y, p=2, dim=dim, eps=eps)
    return (x * y).sum(dim=dim).clamp(min=-1.0, max=1.0)


def compute_sample_cypher_span_representations(hidden_state, token_to_span_map):
    token_weights = token_to_span_map.float()
    span_sums = torch.einsum("ld,ls->sd", hidden_state.float(), token_weights)
    span_lengths = token_weights.sum(dim=0).unsqueeze(-1).clamp(min=1e-5)
    return span_sums / span_lengths


def compute_section_attention_representations(hidden_state, cypher_spans, section_mask):
    token_positions = torch.nonzero(section_mask, as_tuple=False).flatten()
    if token_positions.numel() == 0 or cypher_spans.size(0) == 0:
        return None

    section_hidden = hidden_state[token_positions].float()
    cypher_spans = cypher_spans.float()

    scores = torch.matmul(cypher_spans, section_hidden.transpose(0, 1))
    scores = scores / math.sqrt(hidden_state.size(-1))

    attn_weights = torch.softmax(scores, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
    attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-5)

    return torch.matmul(attn_weights, section_hidden)


def compute_aligned_span_relation_loss_for_section(
    student_hidden_state,
    teacher_hidden_state,
    token_to_span_map,
    span_mask,
    section_mask,
):
    zero = student_hidden_state.new_tensor(0.0)

    loss_num = zero
    loss_den = zero
    for batch_idx in range(student_hidden_state.size(0)):
        cypher_span_lengths = token_to_span_map[batch_idx].float().sum(dim=0)
        valid_span_mask = span_mask[batch_idx] & (cypher_span_lengths > 0)
        if not valid_span_mask.any() or not section_mask[batch_idx].any():
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

        student_section_spans = compute_section_attention_representations(
            student_hidden_state[batch_idx],
            student_cypher_spans,
            section_mask[batch_idx],
        )
        teacher_section_spans = compute_section_attention_representations(
            teacher_hidden_state[batch_idx],
            teacher_cypher_spans,
            section_mask[batch_idx],
        )
        if student_section_spans is None or teacher_section_spans is None:
            continue

        student_rel = _safe_cosine_similarity(student_cypher_spans, student_section_spans, dim=-1)
        teacher_rel = _safe_cosine_similarity(teacher_cypher_spans, teacher_section_spans, dim=-1)
        per_span = (student_rel - teacher_rel).pow(2)
        per_span = torch.nan_to_num(per_span, nan=0.0, posinf=4.0, neginf=0.0).clamp(min=0.0, max=4.0)

        loss_num = loss_num + (per_span * weights).sum()
        loss_den = loss_den + weights.sum()

    if loss_den <= 0:
        return zero
    loss = loss_num / loss_den.clamp(min=1e-5)
    return torch.nan_to_num(loss, nan=0.0, posinf=4.0, neginf=0.0).clamp(min=0.0, max=4.0)


def compute_grounding_losses_for_layer(
    student_hidden_state,
    teacher_hidden_state,
    token_to_span_map,
    span_mask,
    query_mask,
    schema_mask,
):
    query_rel_loss = compute_aligned_span_relation_loss_for_section(
        student_hidden_state,
        teacher_hidden_state,
        token_to_span_map,
        span_mask,
        query_mask,
    )
    schema_rel_loss = compute_aligned_span_relation_loss_for_section(
        student_hidden_state,
        teacher_hidden_state,
        token_to_span_map,
        span_mask,
        schema_mask,
    )
    query_rel_loss = torch.nan_to_num(query_rel_loss, nan=0.0, posinf=4.0, neginf=0.0).clamp(min=0.0, max=4.0)
    schema_rel_loss = torch.nan_to_num(schema_rel_loss, nan=0.0, posinf=4.0, neginf=0.0).clamp(min=0.0, max=4.0)
    return query_rel_loss, schema_rel_loss


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
    token_to_span_map, span_mask = prepare_span_token_map(attention_mask, offsets_mapping, spans_offsets)
    if token_to_span_map is None:
        zero = attention_mask.new_tensor(0.0)
        return zero, zero

    query_mask, schema_mask = build_prompt_section_masks(input_ids, attention_mask, labels, tokenizer)
    if not query_mask.any() and not schema_mask.any():
        zero = attention_mask.new_tensor(0.0)
        return zero, zero

    query_rel_total = attention_mask.new_tensor(0.0)
    schema_rel_total = attention_mask.new_tensor(0.0)
    valid_layers = 0

    for student_idx, teacher_idx in zip(args.student_layer_mapping, args.teacher_layer_mapping):
        student_hidden = student_hidden_states[student_idx]
        teacher_hidden = teacher_hidden_states[teacher_idx]
        if student_hidden is None:
            continue

        query_rel_loss, schema_rel_loss = compute_grounding_losses_for_layer(
            student_hidden,
            teacher_hidden,
            token_to_span_map,
            span_mask,
            query_mask,
            schema_mask,
        )
        query_rel_total += query_rel_loss
        schema_rel_total += schema_rel_loss
        valid_layers += 1

    if valid_layers == 0:
        zero = attention_mask.new_tensor(0.0)
        return zero, zero

    return query_rel_total / valid_layers, schema_rel_total / valid_layers


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

    adaptive_threshold = args.init_threshold if "adaptive" in args.type else -1.0
    prev_avg_loss = 0.0
    replay_buffer = ReplayBuffer(args)

    student_captured_hidden = []
    hook_handles = []

    def capture_hook_fn(module, inputs, output):
        if module.training:
            student_captured_hidden.append(output[0] if isinstance(output, tuple) else output)

    for layer in model.base_model.model.model.layers:
        hook_handles.append(layer.register_forward_hook(capture_hook_fn))

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        alloc_sum = 0.0
        alloc_count = 0

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
                    no_model_batch["label"] = model_batch.pop("no_model_batch")
                    replay_buffer.move_to_memory(model_batch, no_model_batch)
                    model_batch, no_model_batch = replay_buffer.sample()
                    model_batch, no_model_batch = replay_buffer.move_to_device(model_batch, no_model_batch, device)
                elif "adaptive" in args.type and (
                    rand_value < samp_threshold
                    or (rand_value < adaptive_threshold and len(replay_buffer) < args.capacity)
                ):
                    model_batch = student_generator.run_sample(model, gen_data)
                    no_model_batch["label"] = model_batch.pop("no_model_batch")
                    if args.model_type in ["opt"]:
                        model_batch.pop("position_ids")
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
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_model.eval()
                    teacher_outputs = teacher_model(**model_batch, output_hidden_states=True, use_cache=False)
                    teacher_logits = teacher_outputs.logits

                distil_loss = get_distil_loss(args, teacher_logits, no_model_batch, logits)
                distil_loss = torch.nan_to_num(distil_loss, nan=0.0, posinf=100.0, neginf=0.0)
                if "offset_mapping" in no_model_batch and "span_offsets" in no_model_batch:
                    span_query_loss, span_schema_loss = compute_overall_relation_loss(
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

                weighted_span_query_loss = grounding_cfg["w_span_query"] * span_query_loss
                weighted_span_schema_loss = grounding_cfg["w_span_schema"] * span_schema_loss
                weighted_grounding_loss = torch.nan_to_num(
                    weighted_span_query_loss + weighted_span_schema_loss,
                    nan=0.0,
                    posinf=1e4,
                    neginf=0.0,
                )
                weighted_grounding_loss = weighted_grounding_loss.clamp(min=0.0, max=1e4)

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

            reduced_loss = loss.detach().clone()
            dist.all_reduce(reduced_loss, dist.ReduceOp.SUM, group=dp_group)
            global_loss = reduced_loss.item() / dp_world_size

            global_distil_loss = 0.0
            global_grounding_loss = 0.0
            global_span_query_loss = 0.0
            global_span_schema_loss = 0.0
            if teacher_model is not None:
                reduced_distil = distil_loss.detach().clone()
                dist.all_reduce(reduced_distil, dist.ReduceOp.SUM, group=dp_group)
                global_distil_loss = reduced_distil.item() / dp_world_size

                reduced_ground = weighted_grounding_loss.detach().clone()
                dist.all_reduce(reduced_ground, dist.ReduceOp.SUM, group=dp_group)
                global_grounding_loss = reduced_ground.item() / dp_world_size

                reduced_span_query = weighted_span_query_loss.detach().clone()
                dist.all_reduce(reduced_span_query, dist.ReduceOp.SUM, group=dp_group)
                global_span_query_loss = reduced_span_query.item() / dp_world_size

                reduced_span_schema = weighted_span_schema_loss.detach().clone()
                dist.all_reduce(reduced_span_schema, dist.ReduceOp.SUM, group=dp_group)
                global_span_schema_loss = reduced_span_schema.item() / dp_world_size

                total_distil_loss += global_distil_loss
                total_grounding_loss += global_grounding_loss
                total_span_query_loss += global_span_query_loss
                total_span_schema_loss += global_span_schema_loss

            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time
            total_loss += global_loss
            total_time += elapsed_time

            def get_log(log_loss, log_distil_loss, log_ground, log_span_query, log_span_schema, log_time):
                return (
                    "train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | "
                    "loss: {:.4f} | ds_loss: {:.4f} | ground: {:.4f} | "
                    "span_q: {:.4f} | span_s: {:.4f} | lr: {:.4e} | "
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
