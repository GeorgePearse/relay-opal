"""Noise Track 1 (the 37.7M-parameter GPT, one epoch of 4,880 sequences) with one of three arms on one GPU:

    python train.py --arm backprop     # autograd, the record's optimizer (ANVIL II on block matrices, Adam elsewhere)
    python train.py --arm relay        # Relay fills the gradients, OPAL wraps the optimizer with backend="keep"
    python train.py --arm relay-opal --max-cost-usd 15 [--backend claude | --backend mailbox:<folder>]

The model, init, optimizer, rates, warmdown schedule, batch (8 sequences per update) and data order are the Relay
script's; the fixed step count (--steps) scales the schedule down for a short run. Writes config.json, metrics.jsonl,
result.json and final.pt (loadable by qlabs-eng/noise utils/eval.py) to --output.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import SEQ, TRAIN_ROWS, VAL_ROWS, load_rows  # noqa: E402

import relay_opal  # noqa: E402
from relay_opal.models import GPTConfig, NoiseGPTSpec, make_model, make_optimizer  # noqa: E402

SEED = 0
WARMDOWN = 0.3
GOAL = (
    "The 37.7M-parameter GPT of the Noise speedrun on 10M FineWeb tokens, one epoch, gradients estimated forward-only "
    "by Relay; judged by the mean next-token cross-entropy on the fixed validation set after the last update."
)


@torch.no_grad()
def evaluate(model, rows, batch_size=16):
    device = next(model.parameters()).device
    total = torch.zeros(2, dtype=torch.float64, device=device)
    for batch in rows.split(batch_size):
        batch = batch.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = model(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), loss_reduction="sum")
        total[0] += loss.double()
        total[1] += batch[:, 1:].numel()
    return (total[0] / total[1]).item()


def schedulers(opts, steps):
    rate = lambda k: min(1.0, (steps - k) / steps / WARMDOWN)
    return [torch.optim.lr_scheduler.LambdaLR(o, rate) for o in opts]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("logs/relay"))
    parser.add_argument("--arm", choices=("backprop", "relay", "relay-opal"), default="relay")
    parser.add_argument("--steps", type=int, default=0, help="Updates to run (0: the whole epoch, 610).")
    parser.add_argument("--batch", type=int, default=8, help="Sequences per update.")
    parser.add_argument("--micro", type=int, default=1, help="Sequences per Relay estimate call.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--routes", type=int, default=0, help="MLP units per token carrying a move (0: H/4).")
    parser.add_argument("--fidelity-every", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=0, help="Untimed validation every this many updates.")
    parser.add_argument("--backend", default="claude", help="relay-opal: claude, codex or mailbox:<folder>.")
    parser.add_argument("--max-cost-usd", type=float, default=None)
    parser.add_argument("--decisions", type=int, default=None)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--blocking", default="True")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preset", choices=("38m", "tiny"), default="38m", help="tiny: a 2-block width-64 smoke test.")
    parser.add_argument("--synthetic", type=int, default=0, help="Smoke test: this many random rows instead of data.")
    args = parser.parse_args()
    if args.arm == "relay-opal" and args.max_cost_usd is None:
        raise SystemExit("--max-cost-usd is required for the agent arm")

    device = torch.device(args.device)
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    config = GPTConfig() if args.preset == "38m" else GPTConfig(n_layer=2, n_head=1, n_kv_head=1, n_embd=64)
    if args.synthetic:
        g = torch.Generator().manual_seed(SEED)
        training = torch.randint(0, config.vocab_size, (args.synthetic, SEQ), generator=g)
        val_rows = torch.randint(0, config.vocab_size, (16, SEQ), generator=g)
    else:
        training = load_rows(args.data_dir / "train.pt", TRAIN_ROWS)
        val_rows = load_rows(args.data_dir / "val.pt", VAL_ROWS)
    training = training[torch.randperm(len(training), generator=torch.Generator().manual_seed(SEED + 1))]
    steps = args.steps or len(training) // args.batch
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    settings = relay_opal.RelaySettings(routes=args.routes or config.n_embd)  # H/4 of the MLP width, as submitted
    (args.output / "config.json").write_text(
        json.dumps(
            dict(
                arm=args.arm,
                steps=steps,
                batch=args.batch,
                micro=args.micro,
                seed=args.seed,
                model=asdict(config),
                estimator=settings.to_dict() if args.arm != "backprop" else None,
                fidelity_every=args.fidelity_every,
                backend=args.backend if args.arm == "relay-opal" else "keep" if args.arm == "relay" else None,
                max_cost_usd=args.max_cost_usd,
                torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0) if cuda else str(device),
            ),
            indent=2,
        )
        + "\n"
    )

    def build(fresh_seed):
        model = make_model(device, fresh_seed, config)
        opts = make_optimizer(model, steps)
        scheds = schedulers(opts, steps)
        if args.arm == "backprop":
            model.requires_grad_(True)
            return model, opts, scheds, None
        est = relay_opal.RelayEstimator(NoiseGPTSpec(model), settings, seed=fresh_seed)
        kw = (
            dict(backend="keep")
            if args.arm == "relay"
            else dict(
                backend=args.backend,
                max_cost_usd=args.max_cost_usd,
                decisions=args.decisions,
                every=args.every,
                blocking={"True": True, "False": False, "auto": "auto"}[args.blocking],
            )
        )
        loop = relay_opal.wrap(
            opts,
            model,
            est,
            goal=GOAL,
            total_steps=steps,
            scheduler=scheds,
            fidelity_every=args.fidelity_every,
            run_dir=args.output / "opal",
            task_facts=f"One epoch of {len(training):,} sequences of 2,048 tokens, {args.batch} per update, one GPU.",
            summarize=False,
            **kw,
        )
        return model, opts, scheds, loop

    def update(model, opts, scheds, loop, x, y):
        if loop is None:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                loss = model(x, y)
            loss.backward()
            loss = loss.detach()
            for o in opts:
                o.step()
                o.zero_grad(set_to_none=True)
        else:
            loss = loop.step(x, y, micro_batches=args.batch // args.micro)
            loop.opt.zero_grad(set_to_none=True)
        for s in scheds:
            s.step()
        return loss

    # untimed warmup on random tokens (compilation), then training from a fresh model
    noise = torch.randint(0, config.vocab_size, (args.batch, SEQ), generator=torch.Generator().manual_seed(SEED)).to(
        device
    )
    if args.arm != "relay-opal":
        model, opts, scheds, loop = build(args.seed)
        update(model, opts, scheds, loop, noise[:, :-1].contiguous(), noise[:, 1:].contiguous())
        if loop is not None:
            loop.close()
        del model, opts, scheds, loop
        if cuda:
            torch.cuda.empty_cache()
    model, opts, scheds, loop = build(args.seed)

    log = (args.output / "metrics.jsonl").open("a")
    sync()
    start, paused = time.perf_counter(), 0.0
    elapsed = 0.0
    for step in range(1, steps + 1):
        batch = training[(step - 1) * args.batch : step * args.batch].to(device)
        loss = update(model, opts, scheds, loop, batch[:, :-1].contiguous(), batch[:, 1:].contiguous())
        sync()
        elapsed = time.perf_counter() - start - paused
        record = dict(step=step, train_loss=round(float(loss), 6), train_seconds=round(elapsed, 2))
        if loop is not None and loop.fidelity and loop.fidelity[-1]["update"] == step:
            record["fidelity_block_matrices"] = loop.fidelity[-1]["cosine"]["block_matrices"]
        if args.eval_every and step % args.eval_every == 0 and step < steps:
            pause = time.perf_counter()
            record["val_loss"] = evaluate(model, val_rows)
            paused += time.perf_counter() - pause
        print(json.dumps(record), flush=True)
        log.write(json.dumps(record) + "\n")
        log.flush()

    result = dict(
        arm=args.arm,
        val_loss=evaluate(model, val_rows),
        train_seconds=round(elapsed, 2),
        seconds_per_update=round(elapsed / steps, 3),
        steps=steps,
        tokens=steps * args.batch * (SEQ - 1),
        gpu=torch.cuda.get_device_name(0) if cuda else str(device),
    )
    if loop is not None:
        result["fidelity"] = [dict(update=f["update"], cosine=f["cosine"], seconds=f["seconds"]) for f in loop.fidelity]
        result["estimator_history"] = loop.history
        result["opal_run"] = str(loop.opt.run.dir)
        loop.close()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, args.output / "final.pt")
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("fidelity", "estimator_history")}), flush=True)


if __name__ == "__main__":
    main()
