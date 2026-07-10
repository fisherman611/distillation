# Time Compare Scripts

Run these from the repository root with `bash`.

| Table row | Script |
| --- | --- |
| RKL | `scripts/time_compares/train_rkl.sh` |
| RKL w/ `C_rel` | `scripts/time_compares/train_rkl_crel.sh` |
| SFKL | `scripts/time_compares/train_sfkl.sh` |
| SFKL w/ `C_rel` | `scripts/time_compares/train_sfkl_crel.sh` |
| CSD | `scripts/time_compares/train_csd.sh` |
| CSD w/ `C_rel` | `scripts/time_compares/train_csd_crel.sh` |
| DistillM | `scripts/time_compares/train_distillm.sh` |
| DistillM w/ `C_rel` | `scripts/time_compares/train_distillm_crel.sh` |

Defaults use the Qwen 0.6B student, Qwen 4B teacher, `batch_size=8`, `grad_acc=2`, and `log_interval=20`.
RKL, SFKL, and CSD do not enable student generation. DistillM uses `--type adaptive_srkl` with `--student-gen`.

Useful overrides:

```bash
RUN_GPUS=0,1 LOG_INTERVAL=20 bash scripts/time_compares/train_rkl.sh
RUN_GPUS=0,1 W_REL_LOSS=1.0 bash scripts/time_compares/train_rkl_crel.sh
TRAIN_EXTRA_ARGS="--total-iters 100" bash running.sh --mode sequential --filter time_compares --gpus 0,1 --gpus-per-job 2
```

The trainers log:

- `step time` for time/step.
- `train | avg_alloc ... | peak_alloc ...` for memory columns.
