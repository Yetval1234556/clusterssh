"""
epoch_report.py — Verbose per-epoch diagnostic reporter for Bloom training.
Prints a comprehensive summary every N epochs to stdout (captured in SLURM .out file).

Usage:
    from epoch_report import EpochReporter
    reporter = EpochReporter(report_every=5)
    reporter.epoch_start()
    reporter.report(epoch, total_epochs, model, optimizer,
                    train_loss, val_loss, train_acc, val_acc, extra)
"""

import os
import sys
import time
import math
import socket
import datetime
import torch

W = 82   # report width

# ── ANSI colours (show in tail -f, degrade to plain in log viewers) ───────────
_C = {
    "reset":  "\033[0m",  "bold":   "\033[1m",
    "green":  "\033[92m", "yellow": "\033[93m",
    "red":    "\033[91m", "cyan":   "\033[96m",
    "blue":   "\033[94m", "grey":   "\033[90m",
    "white":  "\033[97m", "magenta":"\033[95m",
}
def c(col, txt): return f"{_C.get(col,'')}{txt}{_C['reset']}"
def _bar(val, total=100, width=40, fill="█", empty="░"):
    filled = int(width * min(val, total) / max(total, 1))
    return fill * filled + empty * (width - filled)
def _acc_col(acc):
    if acc >= 90: return "green"
    if acc >= 75: return "yellow"
    return "red"
def _line(char="═"): return char * W
def _box_top():    return "╔" + "═"*(W-2) + "╗"
def _box_bot():    return "╚" + "═"*(W-2) + "╝"
def _box_mid():    return "╠" + "═"*(W-2) + "╣"
def _sec(title):
    pad = (W - len(title) - 4) // 2
    return "╠" + "═"*pad + f"  {title}  " + "═"*(W - pad - len(title) - 4) + "╣"
def _row(text):
    # Pad to W-2 inside box borders, accounting for ANSI escape sequences
    visible = _strip_ansi(text)
    pad = max(0, W - 2 - len(visible))
    return "║ " + text + " "*pad + " ║"

def _strip_ansi(s):
    import re
    return re.sub(r'\033\[[0-9;]*m', '', s)


class EpochReporter:
    def __init__(self, report_every=5):
        self.report_every = report_every
        self.history      = []
        self._job_start   = time.time()
        self._epoch_start = None
        self._best_val_acc  = 0.0
        self._best_val_loss = float("inf")
        self._best_epoch    = 0

    def epoch_start(self):
        self._epoch_start = time.time()

    def report(self, epoch, total_epochs, model, optimizer,
               train_loss, val_loss,
               train_acc=None, val_acc=None, extra=None):
        extra = extra or {}
        epoch_secs = time.time() - (self._epoch_start or time.time())
        total_secs = time.time() - self._job_start

        if val_loss is not None and val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._best_epoch    = epoch
        if val_acc is not None and val_acc > self._best_val_acc:
            self._best_val_acc = val_acc
            if val_loss is None:
                self._best_epoch = epoch

        self.history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_acc": train_acc, "val_acc": val_acc,
            "epoch_secs": epoch_secs, "total_secs": total_secs,
            "lr": extra.get("lr", None),
        })

        if epoch % self.report_every == 0 or epoch == total_epochs:
            self._print_full_report(epoch, total_epochs, model, optimizer, extra)

    # ── Full report ───────────────────────────────────────────────────────────
    def _print_full_report(self, epoch, total_epochs, model, optimizer, extra):
        rec         = self.history[-1]
        total_secs  = rec["total_secs"]
        epoch_secs  = rec["epoch_secs"]
        secs_per_ep = total_secs / epoch if epoch > 0 else 0
        eta_secs    = secs_per_ep * (total_epochs - epoch)
        elapsed_str = str(datetime.timedelta(seconds=int(total_secs)))
        eta_str     = str(datetime.timedelta(seconds=int(eta_secs)))
        now_str     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pct_done    = 100.0 * epoch / total_epochs
        job_id      = os.environ.get("SLURM_JOB_ID", "local")
        hostname    = socket.gethostname()

        train_acc_pct = (rec["train_acc"] or 0) * 100
        val_acc_pct   = (rec["val_acc"]   or 0) * 100

        print()
        print(c("cyan", _box_top()))

        # Header
        header = f"  BLOOM  ·  EPOCH {epoch}/{total_epochs}  ·  {now_str}  ·  job {job_id}"
        print(c("cyan", _row(c("bold", header))))
        print(c("cyan", _row(f"  Host: {hostname}   Elapsed: {elapsed_str}   ETA: {eta_str}")))

        # Progress bar
        prog = _bar(epoch, total_epochs, width=W-20)
        prog_line = f"  [{c('green', prog)}] {pct_done:.1f}%"
        print(c("cyan", _row(prog_line)))
        print(c("cyan", _box_mid()))

        # ── 1. TIMING ─────────────────────────────────────────────────────────
        print(c("cyan", _sec("  ⏱  TIMING")))
        rows = [
            ("This epoch",      f"{epoch_secs:.1f}s"),
            ("Avg per epoch",   f"{secs_per_ep:.1f}s"),
            ("Total elapsed",   elapsed_str),
            ("ETA to finish",   eta_str),
            ("Est. finish at",  (datetime.datetime.now() + datetime.timedelta(seconds=eta_secs)).strftime("%Y-%m-%d %H:%M:%S")),
        ]
        if extra.get("epoch_secs"):
            imgs_train = extra.get("train_samples", 0)
            imgs_val   = extra.get("val_samples", 0)
            if epoch_secs > 0:
                rows.append(("Throughput", f"~{int((imgs_train + imgs_val) / epoch_secs):,} images/s"))
        for label, val in rows:
            print(c("cyan", _row(f"  {label:<28}  {c('white', val)}")))

        # ── 2. LOSS & ACCURACY ─────────────────────────────────────────────────
        print(c("cyan", _sec("  📊  LOSS & ACCURACY")))
        print(c("cyan", _row(f"  {'Metric':<28}  {'Value':>12}  {'Visual':>35}")))
        print(c("cyan", _row(f"  {'─'*28}  {'─'*12}  {'─'*35}")))

        def metric_row(label, value, pct=None, col="white"):
            bar = _bar(pct, 100, width=33) if pct is not None else ""
            return _row(f"  {label:<28}  {c(col, f'{value}'):>12}  {c('grey', bar)}")

        print(c("cyan", metric_row("Train loss",   f"{rec['train_loss']:.8f}")))
        if rec["val_loss"] is not None:
            print(c("cyan", metric_row("Val loss",  f"{rec['val_loss']:.8f}")))
            gap = rec["val_loss"] - rec["train_loss"]
            gap_col = "red" if gap > 0.05 else "green"
            print(c("cyan", _row(f"  {'Val-Train gap':<28}  {c(gap_col, f'{gap:+.8f}'):>12}  "
                                 f"  {c(gap_col, 'OVERFIT WARNING' if gap>0.05 else 'Healthy gap')}")))
        ta_col = _acc_col(train_acc_pct)
        va_col = _acc_col(val_acc_pct)
        print(c("cyan", metric_row("Train accuracy", f"{train_acc_pct:>7.4f}%", train_acc_pct, ta_col)))
        print(c("cyan", metric_row("Val accuracy",   f"{val_acc_pct:>7.4f}%",   val_acc_pct,   va_col)))
        print(c("cyan", metric_row("Best val acc",   f"{self._best_val_acc*100:>7.4f}%  (ep {self._best_epoch})",
                                   self._best_val_acc*100, "green")))
        if rec["lr"] is not None:
            print(c("cyan", _row(f"  {'Learning rate':<28}  {c('yellow', f'{rec[\"lr\"]:.4e}')}")))

        # ── 3. LOSS TREND TABLE ────────────────────────────────────────────────
        if len(self.history) >= 2:
            print(c("cyan", _sec("  📈  LOSS & ACCURACY TREND (all recorded epochs)")))
            hdr = f"  {'Ep':>4}  {'TrainLoss':>12}  {'ValLoss':>12}  {'TrainAcc':>9}  {'ValAcc':>9}  {'Δ ValAcc':>9}  {'Time':>7}  {'LR':>10}"
            print(c("cyan", _row(hdr)))
            print(c("cyan", _row(f"  {'─'*4}  {'─'*12}  {'─'*12}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*7}  {'─'*10}")))
            prev_va = None
            for r in self.history:
                is_cur  = r["epoch"] == epoch
                is_best = r["epoch"] == self._best_epoch
                vl  = f"{r['val_loss']:.8f}"  if r["val_loss"]   is not None else "         —"
                ta  = f"{r['train_acc']*100:.3f}%" if r["train_acc"] is not None else "      —"
                va  = f"{r['val_acc']*100:.3f}%"   if r["val_acc"]   is not None else "      —"
                lr  = f"{r['lr']:.3e}" if r["lr"] is not None else "         —"
                dva = ""
                if r["val_acc"] is not None and prev_va is not None:
                    delta = (r["val_acc"] - prev_va) * 100
                    dva = f"{delta:+.3f}%"
                    dva = c("green" if delta >= 0 else "red", dva)
                if r["val_acc"] is not None:
                    prev_va = r["val_acc"]
                marker = " ◀" if is_cur else ("★" if is_best else "")
                row_col = "bold" if is_cur else ("green" if is_best else "white")
                line = f"  {r['epoch']:>4}  {r['train_loss']:>12.8f}  {vl:>12}  {ta:>9}  {va:>9}  {dva:>9}  {r['epoch_secs']:>6.0f}s  {lr:>10}{marker}"
                print(c("cyan", _row(c(row_col, line))))

        # ── 4. CLASS-BY-CLASS PERFORMANCE ─────────────────────────────────────
        per_class = extra.get("per_class", {})
        if per_class:
            print(c("cyan", _sec("  🔬  CLASS-BY-CLASS PERFORMANCE")))
            print(c("cyan", _row(f"  {'Class':<24}  {'Acc':>8}  {'Bar':<34}  {'n':>6}")))
            print(c("cyan", _row(f"  {'─'*24}  {'─'*8}  {'─'*34}  {'─'*6}")))
            for cls, stats in sorted(per_class.items()):
                acc = stats["acc"]
                bar = _bar(acc, 100, width=32)
                col = _acc_col(acc)
                print(c("cyan", _row(
                    f"  {cls:<24}  {c(col, f'{acc:>7.2f}%')}  {c('grey', bar)}  {stats['total']:>6,}"
                )))
            # Class imbalance warning
            totals = [s["total"] for s in per_class.values()]
            if max(totals) > min(totals) * 3:
                print(c("cyan", _row(c("yellow",
                    f"  ⚠  Class imbalance detected: max={max(totals):,}  min={min(totals):,}  ratio={max(totals)/max(min(totals),1):.1f}x"
                ))))

        # ── 5. LEARNING DYNAMICS ──────────────────────────────────────────────
        print(c("cyan", _sec("  🧠  LEARNING DYNAMICS")))
        if len(self.history) >= 3:
            recent = self.history[-3:]
            loss_trend = [r["train_loss"] for r in recent]
            acc_trend  = [r["val_acc"] for r in recent if r["val_acc"] is not None]
            if len(loss_trend) >= 2:
                loss_velocity = loss_trend[-1] - loss_trend[0]
                loss_accel    = (loss_trend[-1] - loss_trend[-2]) - (loss_trend[-2] - loss_trend[0])
                col = "green" if loss_velocity < 0 else "red"
                print(c("cyan", _row(f"  {'Loss velocity (3ep)':<32}  {c(col, f'{loss_velocity:+.6f}')}")))
                stall = abs(loss_velocity) < 1e-5
                print(c("cyan", _row(f"  {'Loss stalling?':<32}  {c('red','YES — consider LR adjustment') if stall else c('green','No — still moving')}")))
            if len(acc_trend) >= 2:
                acc_velocity = (acc_trend[-1] - acc_trend[0]) * 100
                col = "green" if acc_velocity > 0 else "red"
                print(c("cyan", _row(f"  {'Val acc velocity (3ep)':<32}  {c(col, f'{acc_velocity:+.4f}%')}")))

        # Convergence forecast
        if val_acc_pct < 100 and len(self.history) >= 5 and self.history[-1]["val_acc"] is not None:
            acc_vals = [r["val_acc"] for r in self.history[-5:] if r["val_acc"] is not None]
            if len(acc_vals) >= 2:
                avg_gain = (acc_vals[-1] - acc_vals[0]) / len(acc_vals) * 100  # %/epoch
                if avg_gain > 0:
                    eps_to_95 = max(0, (0.95 - (val_acc_pct/100)) / (avg_gain/100))
                    eps_to_99 = max(0, (0.99 - (val_acc_pct/100)) / (avg_gain/100))
                    print(c("cyan", _row(f"  {'Avg gain/epoch (5ep)':<32}  {c('yellow', f'{avg_gain:+.4f}%')}")))
                    if eps_to_95 < 500:
                        print(c("cyan", _row(f"  {'Est. epochs to 95% val':<32}  ~{int(eps_to_95)} more")))
                    if eps_to_99 < 500:
                        print(c("cyan", _row(f"  {'Est. epochs to 99% val':<32}  ~{int(eps_to_99)} more")))
                else:
                    print(c("cyan", _row(c("yellow", "  ⚠  Val accuracy not improving in last 5 epochs"))))

        # ── 6. OPTIMIZER STATE ────────────────────────────────────────────────
        print(c("cyan", _sec("  ⚙️   OPTIMIZER STATE")))
        print(c("cyan", _row(f"  Type: {type(optimizer).__name__}")))
        for i, pg in enumerate(optimizer.param_groups):
            n_params = sum(p.numel() for p in pg["params"] if p.requires_grad)
            label = "Backbone (fine-tune)" if i == 0 else "Head (our layers)"
            print(c("cyan", _row(f"  Group {i} — {label}  ({n_params:,} params)")))
            print(c("cyan", _row(f"    lr={c('yellow', f'{pg[\"lr\"]:.4e}')}   "
                                 f"wd={pg.get('weight_decay','N/A'):.2e}   "
                                 f"betas={pg.get('betas','N/A')}   "
                                 f"eps={pg.get('eps',0):.2e}")))

        # ── 7. WEIGHT STATISTICS ──────────────────────────────────────────────
        print(c("cyan", _sec("  ⚖️   WEIGHT STATISTICS (trainable layers)")))
        hdr = f"  {'Layer':<48} {'Mean':>9} {'Std':>9} {'L2':>9} {'Min':>9} {'Max':>9} {'%~0':>5}"
        print(c("cyan", _row(hdr)))
        print(c("cyan", _row("  " + "─"*48 + "  " + "─"*9*5 + "─"*5)))
        total_params = total_trainable = total_frozen = 0
        for name, param in model.named_parameters():
            total_params += param.numel()
            if not param.requires_grad:
                total_frozen += param.numel()
                continue
            total_trainable += param.numel()
            with torch.no_grad():
                p    = param.detach().float()
                mean = p.mean().item(); std = p.std().item()
                l2   = p.norm(2).item()
                pmin = p.min().item();  pmax = p.max().item()
                n0   = (p.abs() < 1e-6).float().mean().item() * 100
            short = ("…" + name[-47:]) if len(name) > 48 else name
            print(c("cyan", _row(
                f"  {short:<48} {mean:>9.4f} {std:>9.4f} {l2:>9.2f} {pmin:>9.4f} {pmax:>9.4f} {n0:>4.1f}%"
            )))

        # ── 8. GRADIENT STATISTICS ────────────────────────────────────────────
        print(c("cyan", _sec("  🌊  GRADIENT STATISTICS (trainable layers)")))
        hdr = f"  {'Layer':<48} {'Mean':>10} {'Std':>10} {'Norm':>10} {'Max':>10} {'%NaN':>6}"
        print(c("cyan", _row(hdr)))
        print(c("cyan", _row("  " + "─"*48 + "  " + "─"*10*4 + "─"*6)))
        global_grad_sq = 0.0
        no_grad = []
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if param.grad is None: no_grad.append(name); continue
            with torch.no_grad():
                g     = param.grad.detach().float()
                gmean = g.mean().item(); gstd = g.std().item()
                gnorm = g.norm(2).item(); gmax = g.abs().max().item()
                pnan  = g.isnan().float().mean().item() * 100
                global_grad_sq += gnorm ** 2
            short = ("…" + name[-47:]) if len(name) > 48 else name
            print(c("cyan", _row(
                f"  {short:<48} {gmean:>10.3e} {gstd:>10.3e} {gnorm:>10.4f} {gmax:>10.4f} {pnan:>5.1f}%"
            )))
        global_norm = math.sqrt(global_grad_sq)
        print(c("cyan", _row(f"  {'─'*78}")))
        if global_norm < 1e-7:
            gnorm_disp = c("red", f"{global_norm:.6f}  ← VANISHING GRADIENTS!")
        elif global_norm > 100:
            gnorm_disp = c("red", f"{global_norm:.6f}  ← EXPLODING GRADIENTS!")
        else:
            gnorm_disp = c("green", f"{global_norm:.6f}  ✓ healthy")
        print(c("cyan", _row(f"  {'Global gradient norm':<32}  {gnorm_disp}")))
        if no_grad:
            print(c("cyan", _row(c("yellow", f"  ⚠  {len(no_grad)} layers have no gradient (frozen or unused)"))))

        # ── 9. MODEL HEALTH CHECK ─────────────────────────────────────────────
        print(c("cyan", _sec("  🏥  MODEL HEALTH CHECK")))
        print(c("cyan", _row(f"  {'Total parameters':<32}  {total_params:>14,}")))
        print(c("cyan", _row(f"  {'Trainable (ours)':<32}  {c('green',f\"{total_trainable:>14,}\")}  ({100*total_trainable/total_params:.2f}%)")))
        print(c("cyan", _row(f"  {'Frozen (DinoBloom-G)':<32}  {c('grey',f\"{total_frozen:>14,}\")}  ({100*total_frozen/total_params:.2f}%)")))

        nan_l = [n for n,p in model.named_parameters() if p.data.isnan().any()]
        inf_l = [n for n,p in model.named_parameters() if p.data.isinf().any()]
        dead_l= [(n, (p.data.abs()<1e-8).float().mean().item()*100)
                  for n,p in model.named_parameters()
                  if p.requires_grad and (p.data.abs()<1e-8).float().mean().item() > 0.5]

        print(c("cyan", _row(f"  {'NaN weights':<32}  {c('red','FOUND: '+str(nan_l)) if nan_l else c('green','None ✓')}")))
        print(c("cyan", _row(f"  {'Inf weights':<32}  {c('red','FOUND: '+str(inf_l)) if inf_l else c('green','None ✓')}")))
        print(c("cyan", _row(f"  {'Dead layers (>50% near-zero)':<32}  {c('yellow',str(len(dead_l))+\" layers\") if dead_l else c('green','None ✓')}")))
        for n, pct in dead_l:
            print(c("cyan", _row(c("yellow", f"    → {n}  ({pct:.1f}% near-zero)"))))

        # ── 10. GPU MEMORY ────────────────────────────────────────────────────
        if torch.cuda.is_available():
            print(c("cyan", _sec("  🖥️   GPU MEMORY")))
            for i in range(torch.cuda.device_count()):
                props    = torch.cuda.get_device_properties(i)
                alloc    = torch.cuda.memory_allocated(i)  / 1024**3
                reserved = torch.cuda.memory_reserved(i)   / 1024**3
                total_m  = props.total_memory              / 1024**3
                peak_a   = torch.cuda.max_memory_allocated(i) / 1024**3
                peak_r   = torch.cuda.max_memory_reserved(i)  / 1024**3
                util     = 100.0 * alloc / total_m
                vram_bar = _bar(alloc, total_m, width=32)
                print(c("cyan", _row(f"  GPU {i} — {c('bold', props.name)}")))
                print(c("cyan", _row(f"  [{c('green' if util<80 else 'red', vram_bar)}] {util:.1f}% used")))
                print(c("cyan", _row(f"  Alloc: {alloc:.2f}GB  Reserved: {reserved:.2f}GB  Total: {total_m:.1f}GB")))
                print(c("cyan", _row(f"  Peak alloc: {peak_a:.2f}GB  Peak reserved: {peak_r:.2f}GB")))
                print(c("cyan", _row(f"  SMs: {props.multi_processor_count}   CUDA cap: {props.major}.{props.minor}")))

        # ── 11. DATASET & CONFIG SNAPSHOT ─────────────────────────────────────
        train_n = extra.get("train_samples", "?")
        val_n   = extra.get("val_samples",   "?")
        ncls    = extra.get("num_classes",   "?")
        bs      = extra.get("batch_size",    "?")
        ub      = extra.get("unfreeze_blocks","?")
        if any(x != "?" for x in [train_n, val_n, ncls, bs, ub]):
            print(c("cyan", _sec("  📋  DATASET & CONFIG SNAPSHOT")))
            print(c("cyan", _row(f"  {'Train samples':<28}  {str(train_n):>10}")))
            print(c("cyan", _row(f"  {'Val samples':<28}  {str(val_n):>10}")))
            print(c("cyan", _row(f"  {'Num classes':<28}  {str(ncls):>10}")))
            print(c("cyan", _row(f"  {'Batch size':<28}  {str(bs):>10}")))
            print(c("cyan", _row(f"  {'Unfrozen blocks':<28}  {str(ub):>10}  (of 40 DinoBloom-G blocks)")))
            if train_n != "?" and val_n != "?" and train_n + val_n > 0:
                ratio = 100 * train_n / (train_n + val_n)
                print(c("cyan", _row(f"  {'Train/val split':<28}  {ratio:.1f}% / {100-ratio:.1f}%")))
            slurm_id   = os.environ.get("SLURM_JOB_ID", "—")
            slurm_node = os.environ.get("SLURM_NODELIST", hostname)
            print(c("cyan", _row(f"  {'SLURM job ID':<28}  {slurm_id}")))
            print(c("cyan", _row(f"  {'SLURM node':<28}  {slurm_node}")))

        # ── Footer ────────────────────────────────────────────────────────────
        print(c("cyan", _box_mid()))
        summary = (f"  END OF EPOCH {epoch}/{total_epochs} REPORT  │  "
                   f"Best val acc: {c('green', f'{self._best_val_acc*100:.4f}%')} (epoch {self._best_epoch})  │  "
                   f"ETA: {eta_str}")
        print(c("cyan", _row(summary)))
        print(c("cyan", _box_bot()))
        print()
        sys.stdout.flush()
