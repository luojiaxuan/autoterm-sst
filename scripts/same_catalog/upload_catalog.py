"""Upload a built catalog directory to a public dataset repository and check every file against the local copy.

usage: upload_catalog.py <catalog dir> <repo id> <card file>; HF_TOKEN comes from the environment.
"""

import hashlib
import sys
from pathlib import Path

from huggingface_hub import HfApi


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    catalog, repo, card = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    api = HfApi()
    print("user", api.whoami()["name"], flush=True)
    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo, repo_type="dataset",
                    commit_message="Dataset card")
    info = api.upload_folder(folder_path=str(catalog), path_in_repo="catalog", repo_id=repo, repo_type="dataset",
                             commit_message="Catalog")
    revision = info.oid
    remote = {f.rfilename: f for f in api.dataset_info(repo, revision=revision, files_metadata=True).siblings}
    local = sorted(p for p in catalog.rglob("*") if p.is_file())
    bad = []
    for path in local:
        name = "catalog/" + path.relative_to(catalog).as_posix()
        entry = remote.get(name)
        digest = entry.lfs.sha256 if entry is not None and entry.lfs else None
        if entry is None or entry.size != path.stat().st_size or (digest and digest != sha256(path)):
            bad.append(name)
    print("revision", revision, "files", len(local), "mismatched", bad, flush=True)
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
