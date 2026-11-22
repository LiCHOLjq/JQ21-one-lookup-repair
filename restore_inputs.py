"""Restore upstream files required for graph-level replay of all four datasets."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
from urllib.request import urlopen
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "research_discovery_20261115" / "data"
AMAZON = ROOT / "research_direction_20261119" / "experiments" / "public_data"
CORA_URL = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
CORA_TGZ_SHA256 = "0d4ed463d1627bb7f3e8420effe8f5545fd492ae8f88dab44ce86cee7b26d7e8"
CORA_CACHE_SHA256 = "5656f3726bddb86abd4f82cae57dac48d6dd11785e659f11ab03c4d5e8a81b94"


def check(source, digest, name):
    if hashlib.sha256(source).hexdigest() != digest.lower():
        raise ValueError(f"SHA-256 mismatch for {name}")


def get_source(path, url, digest):
    if path is not None:
        content = path.read_bytes()
    else:
        with urlopen(url, timeout=90) as response:
            content = response.read()
    check(content, digest, str(path or url))
    return content


def restore_cora(local_source):
    target = INPUTS / "cora.npz"
    if target.exists():
        check(target.read_bytes(), CORA_CACHE_SHA256, target)
        return
    raw = get_source(local_source, CORA_URL, CORA_TGZ_SHA256)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        rows = [r.split() for r in tf.extractfile("cora/cora.content").read().decode().splitlines() if r.strip()]
        citations = [r.split() for r in tf.extractfile("cora/cora.cites").read().decode().splitlines() if r.strip()]
    index = {r[0]: i for i, r in enumerate(rows)}
    labels = {name: i for i, name in enumerate(sorted({r[-1] for r in rows}))}
    raw_x = np.array([r[1:-1] for r in rows], dtype=np.float32)
    raw_x /= np.maximum(np.linalg.norm(raw_x, axis=1, keepdims=True), 1e-12)
    pairs = {(index[a], index[b]) for a, b in citations}
    edges = np.asarray(sorted(pairs | {(b, a) for a, b in pairs}), dtype=np.int32)
    arrays = dict(x=np.load(INPUTS / "cora_fitted_svd64.npy", allow_pickle=False),
                  x_original=raw_x,
                  labels=np.asarray([labels[r[-1]] for r in rows], dtype=np.int32).astype(str),
                  ids=np.arange(len(rows), dtype=np.int32).astype(str),
                  edges=edges, directed_edges=edges)
    # Match the frozen NumPy archive's member order and Unix ZIP headers on every OS.
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as output:
        for key, arr in arrays.items():
            info = zipfile.ZipInfo(key + ".npy")
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            with output.open(info, "w", force_zip64=True) as member:
                np.lib.format.write_array(member, arr, allow_pickle=False)
    packed = stream.getvalue()
    check(packed, CORA_CACHE_SHA256, target)
    target.write_bytes(packed)


def restore_amazon(local_source):
    target = AMAZON / "amazon_original.npz"
    meta = json.loads((AMAZON / "provenance.json").read_text(encoding="utf-8"))
    if target.exists():
        check(target.read_bytes(), meta["sha256"], target)
        return
    target.write_bytes(get_source(local_source, meta["source"], meta["sha256"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cora-tgz", type=Path, help="Use an existing upstream tarball")
    parser.add_argument("--amazon-npz", type=Path, help="Use an existing upstream sparse NPZ")
    args = parser.parse_args()
    restore_cora(args.cora_tgz)
    restore_amazon(args.amazon_npz)
    print("Cora and Amazon inputs match the archived study checksums.")
