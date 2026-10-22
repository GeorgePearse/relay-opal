"""Serve the agent arm's mailbox from a machine with a Claude Code login (the broker side of OPAL's mailbox backend).

    python examples/track1/agent_broker.py --host <tunnel host> --port <tunnel port> --box /root/box \
        [--key ~/.ssh/id_ed25519] [--model claude-opus-5-5] [--effort high]

`opal broker` reaches the GPU machine with plain `ssh user@host`; the Modal tunnel has its own port, so this puts an
`ssh` wrapper first on PATH that adds the port, the key and the host-key options, then runs the broker.
"""

import argparse
import os
import stat
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--box", default="/root/box")
    parser.add_argument("--user", default="root")
    parser.add_argument("--key", default="")
    parser.add_argument("--backend", default="claude", choices=("claude", "codex", "keep"))
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--workdir", default="")
    args = parser.parse_args()
    bindir = Path(tempfile.mkdtemp(prefix="relay-opal-ssh-"))
    key = f" -i {Path(args.key).expanduser()}" if args.key else ""
    wrapper = bindir / "ssh"
    wrapper.write_text(
        "#!/bin/sh\n"
        f'exec /usr/bin/ssh -p {args.port}{key} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$@"\n'
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
    cmd = ["opal", "broker", args.box, "--ssh", f"{args.user}@{args.host}", "--backend", args.backend]
    if args.backend != "keep":
        cmd += ["--model", args.model, "--effort", args.effort]
    if args.workdir:
        cmd += ["--workdir", args.workdir]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=False)


if __name__ == "__main__":
    main()
