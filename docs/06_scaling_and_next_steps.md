# 6) Scaling and Next Steps

## Better data

- Increase dataset size.
- Clean noisy rows.
- Add metadata-aware prompts (optional).

## Better model

- Increase depth (`n_layer`) and width (`n_embd`).
- Increase context (`block_size`).
- Keep attention heads dividing embedding size exactly.

## Better training

- Longer schedule.
- Learning-rate warmup + decay.
- Mixed precision (GPU).
- Gradient accumulation for bigger effective batch.

## Better evaluation

- Track perplexity (`exp(loss)`).
- Keep fixed validation split.
- Store sample generations at each eval step.

## Tuning playbook (from current baseline)

Baseline reference is [configs/default.yaml](/home/subbu/Code/try-llm/tiny-llm/configs/default.yaml):
`lr=2.5e-4`, `batch_size=16`, `block_size=192`, `dropout=0.15`, `max_steps=5000`,
`n_layer=6`, `n_embd=384`, `weight_decay=0.1`.

Use this sequence: change one knob, run again, compare train/val loss plus sample quality.

| Symptom | Likely cause | What to try first | If still not fixed |
|---|---|---|---|
| Train + val losses both plateau early (high loss) | Underfitting or not enough optimization budget | Increase `max_steps` (for example `5000 -> 10000`) | Increase capacity (`n_layer` or `n_embd`) and/or modestly raise `block_size` |
| Train loss drops, val loss stalls/rises | Overfitting | Increase `dropout` (for example `0.15 -> 0.2`) | Increase `weight_decay`, add more data, or stop earlier |
| Loss is noisy/unstable between evals | LR too high or small effective batch | Lower `learning_rate` (for example `2.5e-4 -> 2e-4`) | Increase effective batch (bigger `batch_size` or grad accumulation) |
| Repetitive or looping generations | Sampling too sharp or model not trained enough | Increase `temperature` slightly (for example `0.7 -> 0.8`) and/or raise `top_k` | Train longer; if persistent, increase model capacity |
| Outputs forget earlier prompt details | Context window too short | Increase `block_size` (for example `192 -> 256`) | Also increase training steps so model learns to use longer context |
| OOM / too slow | Model/context too large for hardware | Reduce `batch_size` first | Then reduce `block_size`, `n_layer`, or `n_embd`; use mixed precision on GPU |

### Safe default adjustment order

1. **Schedule first**: raise `max_steps`.
2. **Stability second**: tune `learning_rate` and `batch_size`.
3. **Generalization third**: tune `dropout` and `weight_decay`.
4. **Capacity last**: increase `n_layer`, `n_embd`, or `block_size`.

This order helps isolate whether the bottleneck is optimization budget, optimization stability, regularization, or model size.

## Experiment log template (copy/paste)

Use this table to keep each run comparable:

| Run ID | Date | Config changes vs baseline | Steps completed | Final train loss | Final val loss | Approx perplexity (val) | Sample prompt | Sample quality notes | Decision |
|---|---|---|---:|---:|---:|---:|---|---|---|
| run-001 | 2026-10-27 | baseline | 5000 |  |  |  | "Once upon a moonlit night," |  | keep / reject |
| run-002 | 2026-10-28 | `max_steps: 10000` | 10000 |  |  |  | "Once upon a moonlit night," |  | keep / reject |

Quick rules:

- Change one primary knob per run where possible.
- Keep the same validation split and sample prompt for fair comparison.
- Prefer decisions based on both validation loss and sample readability/coherence.

### Optional: auto-generate a table row from training logs

This snippet runs training, saves logs, then prints one Markdown row you can paste into the table above.

```bash
RUN_ID="run-003"
CONFIG_PATH="configs/default.yaml"
CONFIG_CHANGE="max_steps: 10000"
SAMPLE_PROMPT="Once upon a moonlit night,"
LOG_PATH="logs/${RUN_ID}.log"

mkdir -p logs
uv run tiny-llm-train --config "${CONFIG_PATH}" | tee "${LOG_PATH}"

python - "${RUN_ID}" "${CONFIG_CHANGE}" "${SAMPLE_PROMPT}" "${LOG_PATH}" <<'PY'
import math
import re
import sys
from datetime import date
from pathlib import Path

run_id, config_change, sample_prompt, log_path = sys.argv[1:]
text = Path(log_path).read_text(encoding="utf-8")

eval_matches = re.findall(r"\[eval\]\s+step=(\d+)\s+train=([0-9.]+)\s+val=([0-9.]+)", text)
if not eval_matches:
    raise SystemExit("No eval metrics found in log. Check eval_interval/eval_iters settings.")

step_str, train_loss_str, val_loss_str = eval_matches[-1]
step = int(step_str)
train_loss = float(train_loss_str)
val_loss = float(val_loss_str)
val_ppl = math.exp(val_loss)

row = (
    f"| {run_id} | {date.today().isoformat()} | `{config_change}` | {step} | "
    f"{train_loss:.4f} | {val_loss:.4f} | {val_ppl:.2f} | "
    f"\"{sample_prompt}\" |  | keep / reject |"
)
print(row)
PY
```

Tip: after reviewing generation quality, fill in `Sample quality notes` and `Decision`.

### Optional: append runs directly to CSV (for plotting)

This variant appends one row per run to `logs/experiments.csv` and creates the header automatically.

```bash
RUN_ID="run-004"
CONFIG_PATH="configs/default.yaml"
CONFIG_CHANGE="learning_rate: 2e-4"
SAMPLE_PROMPT="Once upon a moonlit night,"
LOG_PATH="logs/${RUN_ID}.log"
CSV_PATH="logs/experiments.csv"

mkdir -p logs
uv run tiny-llm-train --config "${CONFIG_PATH}" | tee "${LOG_PATH}"

python - "${RUN_ID}" "${CONFIG_CHANGE}" "${SAMPLE_PROMPT}" "${LOG_PATH}" "${CSV_PATH}" <<'PY'
import csv
import math
import re
import sys
from datetime import date
from pathlib import Path

run_id, config_change, sample_prompt, log_path, csv_path = sys.argv[1:]
text = Path(log_path).read_text(encoding="utf-8")

eval_matches = re.findall(r"\[eval\]\s+step=(\d+)\s+train=([0-9.]+)\s+val=([0-9.]+)", text)
if not eval_matches:
    raise SystemExit("No eval metrics found in log. Check eval_interval/eval_iters settings.")

step_str, train_loss_str, val_loss_str = eval_matches[-1]
step = int(step_str)
train_loss = float(train_loss_str)
val_loss = float(val_loss_str)
val_ppl = math.exp(val_loss)

csv_file = Path(csv_path)
write_header = not csv_file.exists()
csv_file.parent.mkdir(parents=True, exist_ok=True)

with csv_file.open("a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(
            [
                "run_id",
                "date",
                "config_change",
                "steps_completed",
                "train_loss",
                "val_loss",
                "val_perplexity",
                "sample_prompt",
                "sample_quality_notes",
                "decision",
            ]
        )
    writer.writerow(
        [
            run_id,
            date.today().isoformat(),
            config_change,
            step,
            f"{train_loss:.4f}",
            f"{val_loss:.4f}",
            f"{val_ppl:.2f}",
            sample_prompt,
            "",
            "",
        ]
    )

print(f"Appended row to {csv_file}")
PY
```

Quick plot example (validation loss trend by run order):

```bash
python - <<'PY'
import csv
from pathlib import Path

path = Path("logs/experiments.csv")
rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
for i, row in enumerate(rows, start=1):
    print(f"{i:02d} {row['run_id']}: val_loss={row['val_loss']} val_ppl={row['val_perplexity']}")
PY
```

Matplotlib plot example (saves PNG charts):

```bash
python - <<'PY'
import csv
from pathlib import Path

import matplotlib.pyplot as plt

csv_path = Path("logs/experiments.csv")
rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
if not rows:
    raise SystemExit("No rows found in logs/experiments.csv")

run_ids = [row["run_id"] for row in rows]
val_losses = [float(row["val_loss"]) for row in rows]
val_ppl = [float(row["val_perplexity"]) for row in rows]

out_dir = Path("logs")
out_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 4.5))
plt.plot(run_ids, val_losses, marker="o")
plt.title("Validation Loss by Run")
plt.xlabel("Run ID")
plt.ylabel("Val Loss")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
loss_png = out_dir / "val_loss_curve.png"
plt.savefig(loss_png, dpi=150)
plt.close()

plt.figure(figsize=(8, 4.5))
plt.plot(run_ids, val_ppl, marker="o")
plt.title("Validation Perplexity by Run")
plt.xlabel("Run ID")
plt.ylabel("Val Perplexity")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
ppl_png = out_dir / "val_perplexity_curve.png"
plt.savefig(ppl_png, dpi=150)
plt.close()

print(f"Saved: {loss_png}")
print(f"Saved: {ppl_png}")
PY
```

If matplotlib is not available in your environment, install it once:

```bash
uv add matplotlib
```

## Safety and practical caveats

- Small models can still produce flawed content.
- Never rely on generated text for factual correctness.
- Treat this project as an educational base for further experimentation.
