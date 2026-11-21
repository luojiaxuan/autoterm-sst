"""Copy the 1k slices to a neutral staging dir with node-ready paths and neutral strings."""
import collections, json, re, shutil, sys
from pathlib import Path
import torch

SRC = Path("/mnt/data2/jiaxuanluo/autoterm_taurus_20261024/slices")
DST = Path(sys.argv[1])
NODE_ROOT = "/data/inputs/slices"

def clean(s):
    s = s.replace(str(SRC), NODE_ROOT)
    if s.startswith("/mnt/"):
        s = Path(s).name
    return re.sub(r"(rasst_)?autoterm", "run", s)

def walk(x):
    if isinstance(x, dict):
        return {walk(k) if isinstance(k, str) else k: walk(v) for k, v in x.items()}
    if isinstance(x, list):
        return [walk(v) for v in x]
    if isinstance(x, str):
        return clean(x)
    return x

if DST.exists():
    shutil.rmtree(DST)
sources = collections.Counter()
for src in sorted([*(SRC / "prefix_1k").rglob("*"), *(SRC / "unions").rglob("*")]):
    if src.is_dir():
        continue
    rel = src.relative_to(SRC)
    out = DST / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".json":
        data = json.loads(src.read_text(encoding="utf-8"))
        if src.name == "glossary.json":
            sources.update(e.get("source", "") for e in data)
        out.write_text(json.dumps(walk(data), ensure_ascii=False, indent=1), encoding="utf-8")
    elif src.suffix == ".pt":
        obj = torch.load(src, map_location="cpu", weights_only=False)
        obj = {k: v if isinstance(v, torch.Tensor) else walk(v) for k, v in obj.items()}
        torch.save(obj, out)
    else:
        raise SystemExit(f"unexpected file {src}")
man = walk(json.loads((SRC / "manifest_taurus_1k.json").read_text(encoding="utf-8")))
man["root"] = NODE_ROOT
(DST / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({"files": sum(1 for p in DST.rglob("*") if p.is_file()), "sources": sources.most_common(8)}))
