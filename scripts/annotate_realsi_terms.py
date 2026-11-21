"""Annotate terminology in the RealSI English-to-Chinese talks with GPT through the Codex CLI.

RealSI ships term annotations for only one of its ten en2zh talks. Every talk is
annotated here with one fixed prompt: the model reads the aligned sub-utterances
(English source and human Chinese reference) and lists, per sub-utterance, the terms
whose Chinese rendering needs domain knowledge, as verbatim spans of both sides. A
term is kept only if its source span occurs in the English sub-utterance and its
target span in the Chinese reference. The talk that has human annotations is scored
against them to measure agreement.

The call runs in a clean Codex environment (scripts follow the codex-account-calls
setup): an empty CODEX_HOME holding only the account login, a one-line base
instruction, and every tool and feature off.
"""

import argparse
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DOMAINS = {
    "01-tech": "technology (software systems and computing)", "02-health": "health and medicine",
    "03-edu": "education", "04-fin": "finance and economics", "05-law": "law",
    "06-env": "environment", "07-ent": "entertainment", "08-sci": "science",
    "09-sport": "sports", "10-art": "art",
}
PROMPT = (
    "You are annotating terminology in an English talk and its human Chinese translation. The annotations "
    "will be used to check whether a speech translation system renders the talk's terms correctly. The talk's "
    "domain is {domain}. Below are its aligned sub-utterances as JSON lines, each with an id, the English "
    "source, and the Chinese reference.\n\n"
    "For each sub-utterance, list its terms: domain-specific words or phrases, such as technical concepts, "
    "jargon, and named entities (people, organizations, places, works, products, laws, events), whose correct "
    "Chinese rendering requires domain knowledge or a glossary. A term is a noun phrase or a name, never a "
    "clause. Leave out everyday words, fillers, widely known place names such as countries, continents and "
    "major cities, and general vocabulary that any bilingual speaker would translate correctly. For each term "
    "give src, the English span "
    "copied exactly from the source, and trg, the Chinese span copied exactly from the reference that "
    "translates it. Skip a term whose translation the reference omits. Return every sub-utterance id, with an "
    "empty list when it has no terms.\n\n{lines}"
)
SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["items"],
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["id", "terms"],
        "properties": {"id": {"type": "integer"}, "terms": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["src", "trg"],
            "properties": {"src": {"type": "string"}, "trg": {"type": "string"}}}}}}}},
}
FEATURES_OFF = ["shell_tool", "unified_exec", "apps", "browser_use", "computer_use", "image_generation",
                "multi_agent", "plugins", "memories", "goals", "view_image"]


def sub_utterances(talk):
    out = []
    for seg in talk["segment"]:
        for u in seg["utterance"]:
            out.append({"id": len(out), "start_ms": u["sub_start_time"], "end_ms": u["sub_end_time"],
                        "src": u["sub_src_text"], "trg": u["sub_trg_text"], "human_terms": u.get("term") or []})
    return out


def call(prompt, model, effort, home, workdir):
    schema_file = Path(workdir) / "schema.json"
    schema_file.write_text(json.dumps(SCHEMA))
    out_file = Path(workdir) / "answer.json"
    cmd = ["codex", "exec", "-s", "read-only", "--ephemeral", "--skip-git-repo-check", "-m", model,
           "-c", f"model_reasoning_effort={effort}", "-c", f"model_instructions_file={home}/instructions.txt",
           "-c", "web_search=disabled", "--output-schema", str(schema_file), "-o", str(out_file)]
    for f in FEATURES_OFF:
        cmd += ["-c", f"features.{f}=false"]
    cmd.append("-")
    subprocess.run(cmd, input=prompt, text=True, cwd=workdir, check=True, capture_output=True,
                   env={**os.environ, "CODEX_HOME": home})
    return json.loads(out_file.read_text())


def annotate(path, model, effort, home):
    talk = json.load(open(path, encoding="utf-8"))
    key = path.stem.removeprefix("en2zh-")
    subs = sub_utterances(talk)
    lines = "\n".join(json.dumps({"id": s["id"], "en": s["src"], "zh": s["trg"]}, ensure_ascii=False) for s in subs)
    with tempfile.TemporaryDirectory() as workdir:
        answer = call(PROMPT.format(domain=DOMAINS[key], lines=lines), model, effort, home, workdir)
    by_id = {item["id"]: item["terms"] for item in answer["items"]}
    kept, dropped = [], []
    for s in subs:
        for t in by_id.get(s["id"], []):
            ok = t["src"] and t["trg"] and t["src"] in s["src"] and t["trg"] in s["trg"]
            (kept if ok else dropped).append({"id": s["id"], "start_ms": s["start_ms"], "end_ms": s["end_ms"],
                                              "src": t["src"], "trg": t["trg"]})
    return {"talk": path.stem, "domain": key, "sub_utterances": len(subs), "returned_ids": len(by_id),
            "terms": kept, "dropped_not_verbatim": dropped,
            "human_terms": [{"id": s["id"], "src": t["src"], "trg": t["trg"]} for s in subs for t in s["human_terms"]]}


def agreement(result):
    """Occurrence-level precision and recall against human annotations, matched by sub-utterance and source."""
    human = {(h["id"], h["src"].casefold()) for h in result["human_terms"]}
    auto = {(t["id"], t["src"].casefold()) for t in result["terms"]}
    hit = len(human & auto)
    return {"human": len(human), "auto": len(auto), "matched": hit,
            "precision": round(hit / len(auto), 4) if auto else None,
            "recall": round(hit / len(human), 4) if human else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="gpt-6.2-astra")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    home = tempfile.mkdtemp(prefix="codex-home-")
    os.symlink(Path.home() / ".codex/auth.json", Path(home) / "auth.json")
    (Path(home) / "instructions.txt").write_text("You are a helpful assistant.\n")
    paths = sorted(args.json_dir.glob("en2zh-*.json"))
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda p: annotate(p, args.model, args.effort, home), paths))
    version = subprocess.run(["codex", "--version"], capture_output=True, text=True).stdout.strip()
    payload = {"client": version, "model": args.model, "reasoning_effort": args.effort, "prompt": PROMPT,
               "talks": results, "agreement_with_human": {r["talk"]: agreement(r) for r in results if r["human_terms"]}}
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({r["talk"]: {"kept": len(r["terms"]), "dropped": len(r["dropped_not_verbatim"])} for r in results}))
    print(json.dumps(payload["agreement_with_human"]))


if __name__ == "__main__":
    main()
