"""Run the Track 1 arms on one H100 on Modal (https://modal.com). From the repository root:

    modal run examples/track1/modal_run.py::prepare                     # data, once (CPU, ~15 min)
    modal run examples/track1/modal_run.py --arm backprop --steps 0     # whole epoch
    modal run examples/track1/modal_run.py --arm relay --steps 0 --fidelity-every 50
    modal run examples/track1/modal_run.py --arm relay-opal --steps 0 --max-cost-usd 15 --pubkey ~/.ssh/id_ed25519.pub

Runs write to the `relay-opal-track1` volume under /data/runs/<name>; `modal volume get relay-opal-track1 runs/<name>`
fetches a run. The agent arm needs a Claude Code login the container does not have, so it starts an ssh server,
exposes it through a Modal tunnel, prints the `opal broker` command, and waits for a broker on a machine that has the
login to serve its mailbox folder (OPAL's guide, "GPU machines without an agent"); examples/track1/agent_broker.py
runs that broker with the tunnel's port.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("relay-opal-track1")
volume = modal.Volume.from_name("relay-opal-track1", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("opal-optimizer==0.3.6", "numpy==2.2.6", "tokenizers==0.23.1", "datasets==4.8.4")
    .add_local_python_source("relay_opal")
    .add_local_dir(HERE, remote_path="/root/track1")
)
agent_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("openssh-server")
    .pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("opal-optimizer==0.3.6", "numpy==2.2.6", "tokenizers==0.23.1", "datasets==4.8.4")
    .add_local_python_source("relay_opal")
    .add_local_dir(HERE, remote_path="/root/track1")
)


@app.function(image=image, volumes={"/data": volume}, timeout=4 * 3600, cpu=4, memory=16384)
def prepare():
    subprocess.run(["python", "/root/track1/prepare_data.py", "--output", "/data/track1"], check=True)
    volume.commit()


def _train(name, args):
    out = Path("/data/runs") / name
    t0 = time.time()
    subprocess.run(
        ["python", "/root/track1/train.py", "--data-dir", "/data/track1", "--output", str(out), *args], check=True
    )
    volume.commit()
    result = json.loads((out / "result.json").read_text())
    result["wall_seconds"] = round(time.time() - t0, 1)
    return result


@app.function(image=image, gpu="H100", volumes={"/data": volume}, timeout=3 * 3600)
def train(name: str, args: list):
    return _train(name, args)


@app.function(image=agent_image, gpu="H100", volumes={"/data": volume}, timeout=3 * 3600)
def train_agent(name: str, args: list, pubkey: str):
    """The agent arm: an ssh server for the broker, a mailbox folder, then the run with backend=mailbox."""
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
        return _train(name, [*args, "--backend", f"mailbox:{box}"])


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
    pubkey: str = "",
):
    args = ["--arm", arm, "--steps", str(steps), "--micro", str(micro), "--routes", str(routes), "--every", str(every)]
    args += ["--fidelity-every", str(fidelity_every), "--eval-every", str(eval_every)]
    name = name or f"{arm}-{steps or 'epoch'}-{time.strftime('%Y%m%d-%H%M%S')}"
    if arm == "relay-opal":
        args += ["--max-cost-usd", str(max_cost_usd)]
        if decisions:
            args += ["--decisions", str(decisions)]
        result = train_agent.remote(name, args, Path(pubkey).expanduser().read_text())
    else:
        result = train.remote(name, args)
    print(json.dumps({k: v for k, v in result.items() if k not in ("fidelity", "estimator_history")}, indent=2))
