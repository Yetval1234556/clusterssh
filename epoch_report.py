"""
epoch_report.py — Verbose per-epoch diagnostic reporter for DinoBloom training.
Prints a comprehensive summary every N epochs to stdout (captured in SLURM .out file).

Usage in training script:
    from epoch_report import EpochReporter
    reporter = EpochReporter(report_every=5)

    # At the start of each epoch:
    reporter.epoch_start()

    # At the end of each epoch (after validation):
    reporter.report(
        epoch       = epoch,
        total_epochs= args.epochs,
        model       = model,
        optimizer   = optimizer,
        train_loss  = train_loss,
        val_loss    = val_loss,
        train_acc   = train_acc,   # float 0-1, optional
        val_acc     = val_acc,     # float 0-1, optional
        extra       = {"scheduler_lr": scheduler.get_last_lr()}  # any extra k/v
    )
"""

import sys
import time
import math
import datetime
import torch


class EpochReporter:
    def __init__(self, report_every=5):
        self.report_every = report_every
        self.history = []
        self._job_start = time.time()
        self._epoch_start = None
        self._best_val_loss = float("inf")
        self._best_val_acc = 0.0
        self._best_epoch = 0

    def epoch_start(self):
        """Call at the very beginning of each epoch to track timing."""
        self._epoch_start = time.time()

    def report(self, epoch, total_epochs, model, optimizer,
               train_loss, val_loss,
               train_acc=None, val_acc=None,
               extra=None):
        """Record stats. Prints full report every `report_every` epochs and on the last epoch."""
        epoch_secs = time.time() - (self._epoch_start or time.time())
        total_secs = time.time() - self._job_start

        if val_loss is not None and val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._best_epoch = epoch
        if val_acc is not None and val_acc > self._best_val_acc:
            self._best_val_acc = val_acc

        self.history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "epoch_secs": epoch_secs,
            "total_secs": total_secs,
        })

        if epoch % self.report_every == 0 or epoch == total_epochs:
            self._print_full_report(epoch, total_epochs, model, optimizer, extra or {})

    # ─────────────────────────────────────────────────────────────────────────
    def _print_full_report(self, epoch, total_epochs, model, optimizer, extra):
        W = 80
        BAR  = "=" * W
        bar  = "-" * W
        rec  = self.history[-1]

        def section(title):
            pad = (W - len(title) - 2) // 2
            return "=" * pad + f" {title} " + "=" * (W - pad - len(title) - 2)

        elapsed_str = str(datetime.timedelta(seconds=int(rec["total_secs"])))
        eta_secs    = (rec["total_secs"] / epoch) * (total_epochs - epoch) if epoch > 0 else 0
        eta_str     = str(datetime.timedelta(seconds=int(eta_secs)))
        now_str     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n{BAR}")
        print(f"  EPOCH REPORT  [{epoch}/{total_epochs}]   {now_str}")
        print(BAR)

        # ── TIMING ───────────────────────────────────────────────────────────
        print(section("TIMING"))
        print(f"  This epoch    : {rec['epoch_secs']:.1f}s")
        secs_per_epoch = rec["total_secs"] / epoch if epoch > 0 else 0
        print(f"  Avg/epoch     : {secs_per_epoch:.1f}s")
        print(f"  Total elapsed : {elapsed_str}")
        print(f"  ETA remaining : {eta_str}")
        pct  = 100.0 * epoch / total_epochs
        done = int(pct / 2)
        print(f"  Progress      : [{'#'*done}{'.'*(50-done)}] {pct:.1f}%")

        # ── LOSS & ACCURACY ───────────────────────────────────────────────────
        print(section("LOSS & ACCURACY"))
        print(f"  Train loss    : {rec['train_loss']:.8f}")
        if rec["val_loss"] is not None:
            print(f"  Val   loss    : {rec['val_loss']:.8f}")
            delta = rec["val_loss"] - rec["train_loss"]
            print(f"  Gap (val-train): {delta:+.8f}  {'(overfitting warning)' if delta > 0.05 else '(healthy)'}")
        if rec["train_acc"] is not None:
            print(f"  Train acc     : {rec['train_acc']*100:.4f}%")
        if rec["val_acc"] is not None:
            print(f"  Val   acc     : {rec['val_acc']*100:.4f}%")
        print(f"  Best val loss : {self._best_val_loss:.8f}  (epoch {self._best_epoch})")
        print(f"  Best val acc  : {self._best_val_acc*100:.4f}%")

        # Trend table
        if len(self.history) >= 2:
            print(section("LOSS TREND (all recorded epochs)"))
            print(f"  {'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'Train Acc':>10}  {'Val Acc':>10}  {'Epoch(s)':>8}")
            print(f"  {bar}")
            for r in self.history:
                marker = " <<" if r["epoch"] == epoch else ""
                vl  = f"{r['val_loss']:.8f}"  if r["val_loss"]  is not None else "          N/A"
                ta  = f"{r['train_acc']*100:.3f}%" if r["train_acc"] is not None else "       N/A"
                va  = f"{r['val_acc']*100:.3f}%"   if r["val_acc"]   is not None else "       N/A"
                print(f"  {r['epoch']:>6}  {r['train_loss']:>12.8f}  {vl:>12}  {ta:>10}  {va:>10}  {r['epoch_secs']:>7.1f}s{marker}")

        # ── OPTIMIZER ─────────────────────────────────────────────────────────
        print(section("OPTIMIZER STATE"))
        print(f"  Type          : {type(optimizer).__name__}")
        for i, pg in enumerate(optimizer.param_groups):
            n_params = sum(p.numel() for p in pg["params"] if p.requires_grad)
            print(f"  Param group {i}  ({n_params:,} trainable params)")
            print(f"    lr            : {pg['lr']:.6e}")
            if "weight_decay" in pg:
                print(f"    weight_decay  : {pg['weight_decay']:.6e}")
            if "momentum" in pg:
                print(f"    momentum      : {pg.get('momentum', 'N/A')}")
            if "betas" in pg:
                print(f"    betas         : {pg['betas']}")
            if "eps" in pg:
                print(f"    eps           : {pg['eps']:.2e}")
            if "amsgrad" in pg:
                print(f"    amsgrad       : {pg.get('amsgrad')}")

        # ── WEIGHT STATISTICS ─────────────────────────────────────────────────
        print(section("WEIGHT STATISTICS (per trainable layer)"))
        col = f"  {'Layer':<50} {'Shape':<22} {'Mean':>9} {'Std':>9} {'L2Norm':>9} {'Min':>9} {'Max':>9} {'%~0':>6}"
        print(col)
        print(f"  {bar}")

        total_params  = 0
        total_trainable = 0
        total_frozen  = 0

        for name, param in model.named_parameters():
            total_params += param.numel()
            if not param.requires_grad:
                total_frozen += param.numel()
                continue
            total_trainable += param.numel()
            with torch.no_grad():
                p      = param.detach().float()
                mean   = p.mean().item()
                std    = p.std().item()
                l2     = p.norm(2).item()
                pmin   = p.min().item()
                pmax   = p.max().item()
                near0  = (p.abs() < 1e-6).float().mean().item() * 100
                shape  = str(list(param.shape))
            short = ("…" + name[-(49):]) if len(name) > 50 else name
            print(f"  {short:<50} {shape:<22} {mean:>9.4f} {std:>9.4f} {l2:>9.2f} {pmin:>9.4f} {pmax:>9.4f} {near0:>5.1f}%")

        # ── GRADIENT STATISTICS ───────────────────────────────────────────────
        print(section("GRADIENT STATISTICS (per trainable layer)"))
        col = f"  {'Layer':<50} {'GradMean':>10} {'GradStd':>10} {'GradNorm':>10} {'GradMax':>10} {'%NaN':>7}"
        print(col)
        print(f"  {bar}")

        global_grad_norm_sq = 0.0
        no_grad_layers = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.grad is None:
                no_grad_layers.append(name)
                continue
            with torch.no_grad():
                g     = param.grad.detach().float()
                gmean = g.mean().item()
                gstd  = g.std().item()
                gnorm = g.norm(2).item()
                gmax  = g.abs().max().item()
                pnan  = g.isnan().float().mean().item() * 100
                global_grad_norm_sq += gnorm ** 2
            short = ("…" + name[-(49):]) if len(name) > 50 else name
            print(f"  {short:<50} {gmean:>10.4e} {gstd:>10.4e} {gnorm:>10.4f} {gmax:>10.4f} {pnan:>6.1f}%")

        global_grad_norm = math.sqrt(global_grad_norm_sq)
        print(f"  {bar}")
        print(f"  Global gradient norm  : {global_grad_norm:.8f}")
        if no_grad_layers:
            print(f"  Layers with no grad   : {len(no_grad_layers)}")
            for n in no_grad_layers[:10]:
                print(f"    - {n}")
            if len(no_grad_layers) > 10:
                print(f"    ... and {len(no_grad_layers)-10} more")

        # ── MODEL SUMMARY ─────────────────────────────────────────────────────
        print(section("MODEL SUMMARY"))
        print(f"  Architecture  : {type(model).__name__}")
        print(f"  Total params  : {total_params:,}")
        print(f"  Trainable     : {total_trainable:,}  ({100*total_trainable/total_params:.2f}%)")
        print(f"  Frozen        : {total_frozen:,}  ({100*total_frozen/total_params:.2f}%)")

        # Weight health check
        nan_layers, inf_layers, dead_layers = [], [], []
        for name, param in model.named_parameters():
            if param.data.isnan().any():
                nan_layers.append(name)
            if param.data.isinf().any():
                inf_layers.append(name)
            if param.requires_grad:
                with torch.no_grad():
                    dead_pct = (param.data.abs() < 1e-8).float().mean().item()
                if dead_pct > 0.5:
                    dead_layers.append((name, dead_pct * 100))

        print(f"  NaN weights   : {'NONE (healthy)' if not nan_layers else str(nan_layers)}")
        print(f"  Inf weights   : {'NONE (healthy)' if not inf_layers else str(inf_layers)}")
        if dead_layers:
            print(f"  Dead layers (>50% near-zero weights):")
            for n, pct in dead_layers:
                print(f"    - {n}  ({pct:.1f}% near-zero)")
        else:
            print(f"  Dead neurons  : NONE detected")

        # Gradient health
        if global_grad_norm < 1e-7:
            print(f"  Gradient health : WARNING — global norm {global_grad_norm:.2e} is near zero (vanishing?)")
        elif global_grad_norm > 100.0:
            print(f"  Gradient health : WARNING — global norm {global_grad_norm:.2e} is very large (exploding?)")
        else:
            print(f"  Gradient health : OK  (global norm {global_grad_norm:.4f})")

        # ── GPU MEMORY ────────────────────────────────────────────────────────
        if torch.cuda.is_available():
            print(section("GPU MEMORY"))
            for i in range(torch.cuda.device_count()):
                props    = torch.cuda.get_device_properties(i)
                alloc    = torch.cuda.memory_allocated(i)  / 1024**3
                reserved = torch.cuda.memory_reserved(i)   / 1024**3
                total_m  = props.total_memory              / 1024**3
                util     = 100.0 * alloc / total_m
                peak_a   = torch.cuda.max_memory_allocated(i)  / 1024**3
                peak_r   = torch.cuda.max_memory_reserved(i)   / 1024**3
                print(f"  GPU {i} — {props.name}  (SM {props.major}.{props.minor})")
                print(f"    Current alloc  : {alloc:.3f} GB   reserved: {reserved:.3f} GB")
                print(f"    Peak alloc     : {peak_a:.3f} GB   peak reserved: {peak_r:.3f} GB")
                print(f"    Total VRAM     : {total_m:.1f} GB   utilization: {util:.1f}%")
                print(f"    Multi-proc cnt : {props.multi_processor_count}")

        # ── EXTRA ─────────────────────────────────────────────────────────────
        if extra:
            print(section("EXTRA INFO"))
            for k, v in extra.items():
                print(f"  {k:<30} : {v}")

        print(BAR)
        print(f"  END REPORT  epoch {epoch}/{total_epochs}   best so far: epoch {self._best_epoch}  val_loss={self._best_val_loss:.8f}")
        print(f"{BAR}\n")
        sys.stdout.flush()
