"""Run the Track 1 arms on one H100 on Modal (https://modal.com). From the repository root:

    modal run examples/track1/modal_run.py::prepare                     # data, once (CPU, ~15 min)
    modal run examples/track1/modal_run.py::gpu_tests                   # the CUDA-only tests (PTX seeds)
    modal run examples/track1/modal_run.py --arm backprop --steps 0     # whole epoch
    modal run examples/track1/modal_run.py --arm relay --steps 0 --fidelity-every 50
    modal run examples/track1/modal_run.py --arm relay-opal --steps 0 --max-cost-usd 15 --pubkey ~/.ssh/id_ed25519.pub
    modal run examples/track1/modal_run.py --arm relay-opal-ptx --steps 0 --max-cost-usd 25 --pubkey <key>.pub

Runs write to the `relay-opal-track1` volume under /data/runs/<name>; `modal volume get relay-opal-track1 runs/<name>`
fetches a run. The agent arms need a Claude Code login the container does not have, so they start an ssh server,
expose it through a Modal tunnel, print the `opal broker` command, and wait for a broker on a machine that has the
login to serve the mailbox folder (OPAL's guide, "GPU machines without an agent"); examples/track1/agent_broker.py
runs that broker with the tunnel's port. The PTX arm restarts from its last checkpoint when the process dies, with
the crash text left for the agent.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
PACKAGE = next((p / "relay_opal" for p in HERE.parents if (p / "relay_opal" / "__init__.py").exists()), None)
app = modal.App("relay-opal-track1")
volume = modal.Volume.from_name("relay-opal-track1", create_if_missing=True)
MAX_RESTARTS = 8


def _image(base):
    image = base.pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu128").pip_install(
        "opal-optimizer==0.3.6", "numpy==2.2.6", "tokenizers==0.23.1", "datasets==4.8.4", "pytest==8.4.2"
    )
    if PACKAGE is not None:  # defined locally; inside the container the built image is already attached
        image = image.add_local_dir(PACKAGE, remote_path="/root/relay_opal", ignore=["__pycache__"])
        image = image.add_local_dir(HERE, remote_path="/root/track1", ignore=["__pycache__"])
        image = image.add_local_dir(PACKAGE.parent / "tests", remote_path="/root/tests", ignore=["__pycache__"])
    return image


image = _image(modal.Image.debian_slim(python_version="3.12"))
agent_image = _image(modal.Image.debian_slim(python_version="3.12").apt_install("openssh-server"))


@app.function(image=image, volumes={"/data": volume}, timeout=4 * 3600, cpu=4, memory=16384)
def prepare():
    subprocess.run(["python", "/root/track1/prepare_data.py", "--output", "/data/track1"], check=True)
    volume.commit()


@app.function(image=image, gpu="H100", timeout=1800)
def gpu_tests():
    return subprocess.run(
        ["python", "-m", "pytest", "-q", "/root/tests", "-p", "no:cacheprovider"], cwd="/root"
    ).returncode


def _train(name, args, restarts=0):
    """Run train.py; with restarts > 0, a crash (a dead process) restarts the run from its last checkpoint with the
    crash text left in crash.txt for the training script to hand to the agent."""
    out = Path("/data/runs") / name
    t0 = time.time()
    crashes = []
    for attempt in range(restarts + 1):
        cmd = ["python", "/root/track1/train.py", "--data-dir", "/data/track1", "--output", str(out), *args]
        if attempt:
            cmd += ["--resume"]
        p = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
        if p.returncode == 0:
            break
        tail = p.stderr[-4000:]
        print(tail, flush=True)
        crashes.append(dict(attempt=attempt, returncode=p.returncode, stderr_tail=tail))
        out.mkdir(parents=True, exist_ok=True)
        (out / "crashes.json").write_text(json.dumps(crashes, indent=2) + "\n")
        (out / "crash.txt").write_text(f"attempt {attempt} died with return code {p.returncode}:\n{tail}\n")
        volume.commit()
        if attempt == restarts or not (out / "checkpoints" / "latest.pt").exists():
            raise RuntimeError(f"{name}: died {len(crashes)} time(s), no restart possible; see crashes.json")
    volume.commit()
    result = json.loads((out / "result.json").read_text())
    result["wall_seconds"] = round(time.time() - t0, 1)
    result["crashes"] = len(crashes)
    return result


@app.function(image=image, gpu="H100", volumes={"/data": volume}, timeout=3 * 3600)
def train(name: str, args: list):
    return _train(name, args)


@app.function(image=agent_image, gpu="H100", volumes={"/data": volume}, timeout=3 * 3600)
def train_agent(name: str, args: list, pubkey: str, restarts: int = 0):
    """The agent arms: an ssh server for the broker, a mailbox folder, then the run with backend=mailbox."""
    Path("/run/sshd").mkdir(exist_ok=True)
    subprocess.run(["ssh-keygen", "-A"], check=True)
    ssh = Path("/root/.ssh")
    ssh.mkdir(mode=0o700, exist_ok=True)
    (ssh / "authorized_keys").write_text(pubkey.strip() + "\n")
    os.chmod(ssh / "authorized_keys", 0o600)
    subprocess.Popen(["/usr/sbin/sshd", "-D", "-o", "PermitRootLogin yes", "-o", "PasswordAuthentication no"])
    box = Path("/root/box")
    box.mkdir(exist_ok=True)
    with modal.forward(22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        print(f"BROKER python examples/track1/agent_broker.py --host {host} --port {port} --box {box}", flush=True)
        return _train(name, [*args, "--backend", f"mailbox:{box}"], restarts=restarts)


@app.local_entrypoint()
def main(
    arm: str = "relay",
    steps: int = 0,
    name: str = "",
    fidelity_every: int = 0,
    max_cost_usd: float = 0.0,
    decisions: int = 0,
    every: int = 10,
    micro: int = 1,
    routes: int = 0,
    eval_every: int = 0,
    checkpoint_every: int = 50,
    pubkey: str = "",
):
    args = ["--arm", arm, "--steps", str(steps), "--micro", str(micro), "--routes", str(routes), "--every", str(every)]
    args += ["--fidelity-every", str(fidelity_every), "--eval-every", str(eval_every)]
    name = name or f"{arm}-{steps or 'epoch'}-{time.strftime('%Y%m%d-%H%M%S')}"
    if arm.startswith("relay-opal"):
        args += ["--max-cost-usd", str(max_cost_usd)]
        if decisions:
            args += ["--decisions", str(decisions)]
        if arm == "relay-opal-ptx":
            args += ["--checkpoint-every", str(checkpoint_every)]
        result = train_agent.remote(
            name, args, Path(pubkey).expanduser().read_text(), MAX_RESTARTS if arm.endswith("ptx") else 0
        )
    else:
        result = train.remote(name, args)
    print(json.dumps({k: v for k, v in result.items() if k not in ("fidelity", "estimator_history")}, indent=2))
