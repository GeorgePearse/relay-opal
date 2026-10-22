"""Prepare the fixed data of Noise Track 1 (qlabs-eng/noise utils/prepare_data.py, same split, same pinned hashes).

    python prepare_data.py --output data            # writes data/train.pt and data/val.pt, then checks their hashes
    python prepare_data.py --output data --verify   # only checks existing files

FineWeb sample-10BT in stream order, the first 200,000 documents skipped, then a ~20.5M-token training pool whose
first 4,880 rows are the training set, then the validation documents. Rows are 2,049 tokens. The BPE-4096 tokenizer
is fetched from the noise repository at a pinned commit and checked against its hash.
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import torch

SEQ = 2049
TRAIN_ROWS = 4880
VAL_ROWS = 544
SKIP_DOCS = 200_000
POOL_TOKENS = 20_500_000
VAL_POOL_TOKENS = 1_150_000
EXPECTED = {
    "train.pt": "36ccb57eba6f5474b18acf99edd4a4793187260f7da7cc7053322cce59154bde",
    "val.pt": "5fdb9242ce8ab83e519ab5dd5cf8d64f4d8a99f6e74c137f4ca0f723da5f7437",
}
NOISE_COMMIT = "45b771df4f8f63e9254ae33ad36848dcb28a44ef"
TOKENIZER_URL = f"https://raw.githubusercontent.com/qlabs-eng/noise/{NOISE_COMMIT}/tokenizer/tokenizer.json"
TOKENIZER_SHA256 = "a15168ae52f4a1b1c80b270a5e18a2ce3e038b5e3772a1ad3982044602da0ae7"


def token_hash(rows):
    return hashlib.sha256(rows.to(torch.int32).contiguous().numpy().tobytes()).hexdigest()


def load_rows(path, rows):
    """Load a fixed split as int64 [rows, 2049]; refuses anything that is not the pinned data."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, torch.Tensor) or tuple(data.shape) != (rows, SEQ):
        raise ValueError(f"{path}: expected a [{rows}, {SEQ}] token tensor")
    if token_hash(data) != EXPECTED[Path(path).name]:
        raise ValueError(f"{path}: token hash mismatch; rerun prepare_data.py")
    return data.long()


def fetch_tokenizer(out):
    path = out / "tokenizer.json"
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        out.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(TOKENIZER_URL).read())
    if hashlib.sha256(path.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise SystemExit(f"{path} is not the fixed BPE-4096 tokenizer")
    return path


def encode_split(documents, tokenizer, min_tokens):
    tokens = []
    end = tokenizer.token_to_id("<|endoftext|>")
    while len(tokens) < min_tokens:
        tokens.append(end)
        tokens.extend(tokenizer.encode(next(documents)["text"]).ids)
    usable = len(tokens) // SEQ * SEQ
    return torch.tensor(tokens[:usable], dtype=torch.int32).view(-1, SEQ)


def verify(out):
    ok = True
    for name, rows in (("train.pt", TRAIN_ROWS), ("val.pt", VAL_ROWS)):
        data = torch.load(out / name, map_location="cpu", weights_only=True)
        digest = token_hash(data)
        status = "OK" if digest == EXPECTED[name] else "MISMATCH"
        ok &= status == "OK" and tuple(data.shape) == (rows, SEQ)
        print(f"{out / name}: {tuple(data.shape)} sha256 {digest} [{status}]", flush=True)
    if not ok:
        raise SystemExit("data does not match the pinned hashes")


def prepare(out):
    from datasets import load_dataset
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(fetch_tokenizer(out)))
    corpus = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    documents = iter(corpus)
    for _ in range(SKIP_DOCS):
        next(documents)
    train = encode_split(documents, tokenizer, POOL_TOKENS)[:TRAIN_ROWS].clone()
    val = encode_split(documents, tokenizer, VAL_POOL_TOKENS)[:VAL_ROWS].clone()
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(corpus="HuggingFaceFW/fineweb", subset="sample-10BT", skipped_documents=SKIP_DOCS, files={})
    for name, rows in (("train.pt", train), ("val.pt", val)):
        torch.save(rows, out / name)
        manifest["files"][name] = dict(
            rows=len(rows), prediction_tokens=len(rows) * (SEQ - 1), token_sha256=token_hash(rows)
        )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    verify(args.output) if args.verify else prepare(args.output)


if __name__ == "__main__":
    main()
