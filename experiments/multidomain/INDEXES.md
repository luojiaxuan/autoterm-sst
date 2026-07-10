# MaxSim index provenance

MaxSim indexes are binary (`*.pt`, ~40–180 MB each) and are **not** committed
(git-ignored; the runtime keeps them on the demo/eval host and reaches them via
the term-memory manifest). They are fully reproducible from the glossaries in
`glossaries/` (finance/legal) plus the demo nlp/medicine slices.

## Cluster locations (as built for this exploration)

aries-local (used by the 4-talk runs):
`/mnt/data3/jiaxuanluo/local_cache/term_memory/indexes/<name>/en-zh/maxsim.pt`

taurus runtime (canonical):
`/mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/term_memory/indexes/<name>/en-zh/maxsim.pt`

| index `<name>` | terms | glossary source |
|---|---:|---|
| `demo_nlp_gs10k` | 9,933 | curated ACL gold union (238 GT + wiki fillers) |
| `demo_nlp_gs1k` | 1,000 | same GT, fillers capped at 1000 |
| `demo_medicine_gs10k` | 9,996 | curated medicine gold union + session names |
| `demo_medicine_gs1k` | 1,000 | same gold, fillers capped at 1000 |
| `finance_wiki_12k` | 12,000 | `glossaries/finance_wiki_12k.json` (Wikidata) |
| `legal_wiki_12k` | 12,000 | `glossaries/legal_wiki_12k.json` (Wikidata) |
| `merged_4domain_44k` | 41,667 | dedup(nlp10k ∪ medicine10k ∪ finance12k ∪ legal12k) |

The index embeds English source terms; per-language target strings
(`target_translations.{zh,ja,de}`) are resolved from the glossary at prompt
time, so one `en-zh` index serves all three target languages.

## Rebuild (one line per glossary)

```bash
CKPT=checkpoints/retriever/rasst-hn1024.pt
python /path/to/RASST/retriever/build_maxsim_index.py \
  --model-path $CKPT \
  --glossary-path glossaries/finance_wiki_12k.json \
  --output-path <runtime>/indexes/finance_wiki_12k/en-zh/maxsim.pt --device cuda:0
```

The merged glossary is rebuilt by deduplicating the four domain glossaries by
lowercased term (nlp/medicine gold first), then indexed the same way.
