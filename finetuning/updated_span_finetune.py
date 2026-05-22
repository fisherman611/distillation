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


def get_grounding_loss_config(args):
    w_rel = float(getattr(args, "w_rel_loss", 0.0))
    if w_rel == 0.0:
        w_rel = float(getattr(args, "w_span_loss", 1.0))
    if (not math.isfinite(w_rel)) or w_rel < 0.0:
        w_rel = 1.0
    return {"w_rel": min(w_rel, 1e4)}


def build_source_token_mask(attention_mask, labels):
    valid_token_mask = attention_mask.bool()
    ignored_mask = (labels == -100) & valid_token_mask
    target_mask = (labels != -100) & valid_token_mask

    source_mask = ignored_mask.clone()
    has_target = target_mask.any(dim=-1)
    if has_target.any():
        first_target_idx = target_mask.long().argmax(dim=-1)
        source_mask[has_target, first_target_idx[has_target]] = True

    no_target = (~has_target) & valid_token_mask.any(dim=-1)
    if no_target.any():
        source_mask[no_target] = valid_token_mask[no_target]

    return source_mask


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


def compute_span_mean_representations(hidden_state, token_to_span_map, span_mask):
    token_weights = token_to_span_map.float()
    span_sums = torch.einsum("bld,bls->bsd", hidden_state.float(), token_weights)
    span_counts = token_weights.sum(dim=1).unsqueeze(-1).clamp(min=1e-5)
    span_repr = span_sums / span_counts
    span_repr = span_repr * span_mask.unsqueeze(-1).float()
    return span_repr


def compute_query_conditioned_representations(hidden_state, span_repr, span_mask, source_mask):
    hidden_state = hidden_state.float()
    span_repr = span_repr.float()

    scores = torch.matmul(span_repr, hidden_state.transpose(1, 2))
    scores = scores / math.sqrt(hidden_state.size(-1))
    # Use a finite large negative number to keep softmax numerically stable.
    scores = scores.masked_fill(~source_mask.unsqueeze(1), -1e4)

    attn_weights = torch.softmax(scores, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
    attn_weights = attn_weights * source_mask.unsqueeze(1).float()
    attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-5)
    attn_weights = attn_weights * span_mask.unsqueeze(-1).float()

    query_repr = torch.matmul(attn_weights, hidden_state)
    query_repr = query_repr * span_mask.unsqueeze(-1).float()

    return query_repr


def _safe_cosine_similarity(x, y, dim=-1, eps=1e-6):
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    y = torch.nan_to_num(y.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    x = F.normalize(x, p=2, dim=dim, eps=eps)
    y = F.normalize(y, p=2, dim=dim, eps=eps)
    return (x * y).sum(dim=dim).clamp(min=-1.0, max=1.0)


def compute_span_query_relation_loss(student_span, student_query, teacher_span, teacher_query, span_mask, span_lengths):
    student_rel = _safe_cosine_similarity(student_span, student_query, dim=-1)
    teacher_rel = _safe_cosine_similarity(teacher_span, teacher_query, dim=-1)
    per_span = (student_rel - teacher_rel).pow(2)
    per_span = torch.nan_to_num(per_span, nan=0.0, posinf=4.0, neginf=0.0).clamp(min=0.0, max=4.0)
    weights = span_lengths * span_mask.float()
    loss = (per_span * weights).sum() / weights.sum().clamp(min=1e-5)
    return torch.nan_to_num(loss, nan=0.0, posinf=4.0, neginf=0.0).clamp(min=0.0, max=4.0)


def compute_grounding_losses_for_layer(
    student_hidden_state,
    teacher_hidden_state,
    token_to_span_map,
    span_mask,
    source_mask,
):
    valid_sample_mask = span_mask.any(dim=-1) & source_mask.any(dim=-1)
    if not valid_sample_mask.any():
        zero = student_hidden_state.new_tensor(0.0)
        return zero

    student_hidden_state = student_hidden_state[valid_sample_mask]
    teacher_hidden_state = teacher_hidden_state[valid_sample_mask]
    token_to_span_map = token_to_span_map[valid_sample_mask]
    span_mask = span_mask[valid_sample_mask]
    source_mask = source_mask[valid_sample_mask]

    span_lengths = token_to_span_map.float().sum(dim=1)
    student_span = compute_span_mean_representations(student_hidden_state, token_to_span_map, span_mask)
    teacher_span = compute_span_mean_representations(teacher_hidden_state, token_to_span_map, span_mask)

    student_query = compute_query_conditioned_representations(student_hidden_state, student_span, span_mask, source_mask)
    teacher_query = compute_query_conditioned_representations(teacher_hidden_state, teacher_span, span_mask, source_mask)
    rel_loss = compute_span_query_relation_loss(
        student_span, student_query, teacher_span, teacher_query, span_mask, span_lengths
    )

    rel_loss = torch.nan_to_num(rel_loss, nan=0.0, posinf=4.0, neginf=0.0).clamp(min=0.0, max=4.0)
    return rel_loss


def compute_overall_relation_loss(
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
        return zero

    source_mask = build_source_token_mask(attention_mask, labels)
    if not source_mask.any():
        zero = attention_mask.new_tensor(0.0)
        return zero

    rel_total = attention_mask.new_tensor(0.0)
    valid_layers = 0

    for student_idx, teacher_idx in zip(args.student_layer_mapping, args.teacher_layer_mapping):
        student_hidden = student_hidden_states[student_idx]
        teacher_hidden = teacher_hidden_states[teacher_idx]
        if student_hidden is None:
            continue

        rel_loss = compute_grounding_losses_for_layer(student_hidden, teacher_hidden, token_to_span_map, span_mask, source_mask)
        rel_total += rel_loss
        valid_layers += 1

    if valid_layers == 0:
        zero = attention_mask.new_tensor(0.0)
        return zero

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
    total_rel_loss = 0.0

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
            weighted_rel_loss = logits.new_tensor(0.0)
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_model.eval()
                    teacher_outputs = teacher_model(**model_batch, output_hidden_states=True, use_cache=False)
                    teacher_logits = teacher_outputs.logits

                distil_loss = get_distil_loss(args, teacher_logits, no_model_batch, logits)
                distil_loss = torch.nan_to_num(distil_loss, nan=0.0, posinf=100.0, neginf=0.0)
                rel_loss = compute_overall_relation_loss(
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
            global_rel_loss = 0.0
            if teacher_model is not None:
                reduced_distil = distil_loss.detach().clone()
                dist.all_reduce(reduced_distil, dist.ReduceOp.SUM, group=dp_group)
                global_distil_loss = reduced_distil.item() / dp_world_size

                reduced_ground = weighted_grounding_loss.detach().clone()
                dist.all_reduce(reduced_ground, dist.ReduceOp.SUM, group=dp_group)
                global_grounding_loss = reduced_ground.item() / dp_world_size

                reduced_rel = weighted_rel_loss.detach().clone()
                dist.all_reduce(reduced_rel, dist.ReduceOp.SUM, group=dp_group)
                global_rel_loss = reduced_rel.item() / dp_world_size

                total_distil_loss += global_distil_loss
                total_grounding_loss += global_grounding_loss
                total_rel_loss += global_rel_loss

            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time
            total_loss += global_loss
            total_time += elapsed_time

            def get_log(log_loss, log_distil_loss, log_ground, log_rel, log_time):
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
                    log_loss,
                    log_distil_loss,
                    log_ground,
                    log_rel,
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
                            global_rel_loss,
                            0.0,
                        )
                    )

            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                denom = args.log_interval * args.gradient_accumulation_steps
                log_str = get_log(
                    total_loss / denom,
                    total_distil_loss / denom,
                    total_grounding_loss / denom,
                    total_rel_loss / denom,
                    total_time / args.log_interval,
                )
                print_rank("*" * 100)
                print_rank(log_str)
                print_rank(args.save)
                print_rank("*" * 100)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                total_loss, total_distil_loss, total_grounding_loss, total_time = 0.0, 0.0, 0.0, 0.0
                total_rel_loss = 0.0

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
