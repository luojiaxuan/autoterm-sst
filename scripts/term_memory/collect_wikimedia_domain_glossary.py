#!/usr/bin/env python3
"""Collect auditable bilingual domain candidates from Wikidata and Wikipedia.

Each domain starts with entities whose ``instance of`` / ``subclass of`` path
reaches a Wikidata root concept.  When that strict RDF slice has fewer than the
requested number of bilingual labels, the collector traverses the matching
English Wikipedia category tree and keeps pages with a Chinese language link.
Every output row retains either a QID or its category path.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


WIKIDATA_ROOTS: Dict[str, str] = {
    "education": "Q8434",
    "finance": "Q43015",
    "legal": "Q7748",
    "environment": "Q43619",
    "entertainment": "Q173799",
    "science": "Q336",
    "sports": "Q349",
    "art": "Q735",
}

WIKIDATA_EXACT_P31_TYPES: Dict[str, Tuple[str, ...]] = {
    "education": (
        "Q2385804",  # educational institution
        "Q3914",  # school
        "Q3918",  # university
        "Q11862829",  # academic discipline
        "Q189004",  # college
    ),
    "science": (
        "Q11173",  # chemical compound
        "Q16521",  # taxon
        "Q6999",  # astronomical object
        "Q7946",  # mineral
        "Q17737",  # theory
        "Q11862829",  # academic discipline
    ),
}

WIKIDATA_SKIP_ABSTRACT_ROOT = {"science"}

WIKIPEDIA_ROOT_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "education": ("Education",),
    "finance": ("Finance",),
    "legal": ("Law",),
    "environment": ("Environment",),
    "entertainment": ("Entertainment",),
    "science": ("Science",),
    "sports": ("Sports",),
    "art": ("Arts",),
}

WIKIPEDIA_DEEPCAT_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "education": (
        "Pedagogy",
        "Teaching",
        "Learning",
        "Curricula",
        "Educational technology",
        "Educational psychology",
        "Schools",
        "Higher education",
        "Academic degrees",
        "Educational assessment and evaluation",
        "Special education",
        "Vocational education",
    ),
    "finance": (
        "Banking",
        "Investment",
        "Financial markets",
        "Accounting",
        "Corporate finance",
        "Financial services",
        "Monetary policy",
        "Insurance",
        "Securities (finance)",
        "Financial risk",
    ),
    "legal": (
        "Legal concepts",
        "Legal doctrines and principles",
        "Legal terminology",
        "Courts",
        "Legislation",
        "Contract law",
        "Criminal law",
        "Civil law",
        "International law",
        "Intellectual property law",
        "Legal professions",
        "Human rights",
    ),
    "environment": (
        "Environmental science",
        "Ecology",
        "Conservation",
        "Pollution",
        "Climate change",
        "Environmental law",
        "Natural disasters",
        "Renewable energy",
        "Biodiversity",
        "Waste management",
    ),
    "entertainment": (
        "Film",
        "Television",
        "Music",
        "Video games",
        "Theatre",
        "Radio",
        "Performing arts",
        "Mass media",
    ),
    "science": (
        "Biology",
        "Chemistry",
        "Physics",
        "Astronomy",
        "Earth sciences",
        "Scientific method",
        "Scientific instruments",
        "Laboratories",
    ),
    "sports": (
        "Association football",
        "Basketball",
        "Baseball",
        "Tennis",
        "Cricket",
        "Athletics (sport)",
        "Sports competitions",
        "Sports terminology",
    ),
    "art": (
        "Visual arts",
        "Painting",
        "Sculpture",
        "Architecture",
        "Art history",
        "Museums",
        "Decorative arts",
        "Photography",
    ),
}

_BAD_TITLE_PREFIXES = (
    "list of ",
    "lists of ",
    "outline of ",
    "index of ",
    "timeline of ",
    "glossary of ",
)
_BAD_CATEGORY_MARKERS = (
    "wikipedia",
    "articles ",
    " lists",
    "lists of",
    "outlines of",
    "portals",
    "templates",
    "stubs",
    "categories",
    "main topic classifications",
    "disambiguation",
)
_YEAR_TITLE = re.compile(r"^[12][0-9]{3}(?: in .+)?$")


def normalized_term(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def stable_hash(*values: str) -> str:
    payload = "\0".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def acceptable_title(title: str) -> bool:
    clean = " ".join(title.split())
    lowered = clean.casefold()
    if len(clean) < 2 or len(clean) > 160:
        return False
    if any(lowered.startswith(prefix) for prefix in _BAD_TITLE_PREFIXES):
        return False
    if "(disambiguation)" in lowered or _YEAR_TITLE.fullmatch(lowered):
        return False
    return any(character.isalpha() for character in clean)


def traversable_category(title: str) -> bool:
    lowered = title.casefold()
    return not any(marker in lowered for marker in _BAD_CATEGORY_MARKERS)


class WikimediaClient:
    def __init__(
        self,
        *,
        user_agent: str,
        timeout_s: float,
        retries: int,
        min_interval_s: float,
        cache_path: Path,
    ) -> None:
        self.user_agent = user_agent
        self.timeout_s = timeout_s
        self.retries = retries
        self.min_interval_s = min_interval_s
        self._request_lock = threading.Lock()
        self._next_request_at = 0.0
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = sqlite3.connect(cache_path, check_same_thread=False)
        self._cache.execute(
            "CREATE TABLE IF NOT EXISTS responses (cache_key TEXT PRIMARY KEY, body TEXT NOT NULL)"
        )
        self._cache.commit()
        self._cache_lock = threading.Lock()

    def _cached(self, cache_key: str) -> Dict[str, Any] | None:
        with self._cache_lock:
            row = self._cache.execute(
                "SELECT body FROM responses WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def _store(self, cache_key: str, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._cache_lock:
            self._cache.execute(
                "INSERT OR REPLACE INTO responses(cache_key, body) VALUES (?, ?)",
                (cache_key, body),
            )
            self._cache.commit()

    def _rate_limit(self) -> None:
        with self._request_lock:
            now = time.monotonic()
            wait_s = max(0.0, self._next_request_at - now)
            if wait_s:
                time.sleep(wait_s)
            self._next_request_at = time.monotonic() + self.min_interval_s

    def get_json(self, endpoint: str, params: Mapping[str, Any]) -> Dict[str, Any]:
        url = endpoint + "?" + urllib.parse.urlencode(params)
        cache_key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        cached = self._cached(cache_key)
        if cached is not None:
            return cached
        for attempt in range(self.retries + 1):
            self._rate_limit()
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": self.user_agent,
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    payload = json.load(response)
                self._store(cache_key, payload)
                return payload
            except urllib.error.HTTPError as exc:
                if attempt >= self.retries:
                    raise
                retry_after = exc.headers.get("Retry-After", "") if exc.headers else ""
                try:
                    wait_s = float(retry_after)
                except ValueError:
                    wait_s = 0.0
                if exc.code == 429:
                    wait_s = max(wait_s, 5.0 * (attempt + 1))
                else:
                    wait_s = max(wait_s, 1.5 * (2**attempt))
                time.sleep(min(30.0, wait_s))
            except (
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                http.client.IncompleteRead,
            ):
                if attempt >= self.retries:
                    raise
                time.sleep(min(10.0, 1.5 * (2**attempt)))
        raise RuntimeError("unreachable")


def _rdf_query(root_qid: str, limit: int) -> str:
    return f"""
SELECT DISTINCT ?item ?itemLabel ?zhLabel WHERE {{
  {{ ?item wdt:P31/wdt:P279* wd:{root_qid} . }}
  UNION
  {{ ?item wdt:P279+ wd:{root_qid} . }}
  ?item rdfs:label ?itemLabel .
  FILTER(LANG(?itemLabel) = \"en\")
  ?item rdfs:label ?zhLabel .
  FILTER(LANG(?zhLabel) = \"zh\")
}}
LIMIT {limit}
""".strip()


def collect_rdf_rows(
    client: WikimediaClient,
    *,
    domain: str,
    root_qid: str,
    limit: int,
) -> Tuple[List[Dict[str, Any]], str]:
    query = _rdf_query(root_qid, limit)
    payload = client.get_json(
        "https://query.wikidata.org/sparql",
        {"query": query, "format": "json"},
    )
    rows: Dict[str, Dict[str, Any]] = {}
    for binding in payload.get("results", {}).get("bindings", []):
        term = str(binding.get("itemLabel", {}).get("value") or "").strip()
        target = str(binding.get("zhLabel", {}).get("value") or "").strip()
        item_url = str(binding.get("item", {}).get("value") or "")
        qid = item_url.rsplit("/", 1)[-1]
        key = normalized_term(term)
        if not key or not target or not acceptable_title(term) or not qid.startswith("Q"):
            continue
        rows.setdefault(
            key,
            {
                "term": term,
                "term_key": key,
                "target_translations": {"zh": target},
                "source": "wikidata_p31_p279",
                "wikidata_qid": qid,
                "domain": domain,
                "domain_root_qid": root_qid,
                "rdf_path": "P31/P279*|P279+",
            },
        )
    result = list(rows.values())
    result.sort(key=lambda row: stable_hash(domain, str(row["wikidata_qid"])))
    return result, hashlib.sha256(query.encode("utf-8")).hexdigest()


def collect_exact_p31_rows(
    client: WikimediaClient,
    *,
    domain: str,
    type_qids: Sequence[str],
    per_type_limit: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: Dict[str, Dict[str, Any]] = {}
    query_stats: List[Dict[str, Any]] = []
    for type_qid in type_qids:
        query = f"""
SELECT DISTINCT ?item ?itemLabel ?zhLabel WHERE {{
  ?item wdt:P31 wd:{type_qid} .
  ?item rdfs:label ?itemLabel .
  FILTER(LANG(?itemLabel) = \"en\")
  ?item rdfs:label ?zhLabel .
  FILTER(LANG(?zhLabel) = \"zh\")
}}
LIMIT {per_type_limit}
""".strip()
        try:
            payload = client.get_json(
                "https://query.wikidata.org/sparql",
                {"query": query, "format": "json"},
            )
        except (OSError, ValueError, TimeoutError) as exc:
            query_stats.append(
                {
                    "type_qid": type_qid,
                    "rows_added": 0,
                    "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        before = len(rows)
        for binding in payload.get("results", {}).get("bindings", []):
            term = str(binding.get("itemLabel", {}).get("value") or "").strip()
            target = str(binding.get("zhLabel", {}).get("value") or "").strip()
            qid = str(binding.get("item", {}).get("value") or "").rsplit("/", 1)[-1]
            key = normalized_term(term)
            if not key or not target or not qid.startswith("Q") or not acceptable_title(term):
                continue
            rows.setdefault(
                key,
                {
                    "term": term,
                    "term_key": key,
                    "target_translations": {"zh": target},
                    "source": "wikidata_exact_p31",
                    "wikidata_qid": qid,
                    "domain": domain,
                    "wikidata_type_qid": type_qid,
                    "rdf_path": "P31",
                },
            )
        query_stats.append(
            {
                "type_qid": type_qid,
                "rows_added": len(rows) - before,
                "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            }
        )
    result = list(rows.values())
    result.sort(key=lambda row: stable_hash(domain, str(row["wikidata_qid"])))
    return result, query_stats


@dataclass(frozen=True)
class PageCandidate:
    pageid: int
    title: str
    depth: int
    category_path: Tuple[str, ...]


def _category_members(
    client: WikimediaClient,
    category_title: str,
    *,
    max_members: int,
) -> List[Dict[str, Any]]:
    members: List[Dict[str, Any]] = []
    continuation = ""
    while len(members) < max_members:
        params: Dict[str, Any] = {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "list": "categorymembers",
            "cmtitle": category_title,
            "cmnamespace": "0|14",
            "cmtype": "page|subcat",
            "cmlimit": "max",
        }
        if continuation:
            params["cmcontinue"] = continuation
        payload = client.get_json("https://en.wikipedia.org/w/api.php", params)
        batch = payload.get("query", {}).get("categorymembers", [])
        members.extend(item for item in batch if isinstance(item, dict))
        continuation = str(payload.get("continue", {}).get("cmcontinue") or "")
        if not continuation or not batch:
            break
    return members[:max_members]


def _direct_subcategories(
    client: WikimediaClient,
    category_title: str,
) -> List[str]:
    output: List[str] = []
    continuation = ""
    while True:
        params: Dict[str, Any] = {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "list": "categorymembers",
            "cmtitle": category_title,
            "cmnamespace": 14,
            "cmtype": "subcat",
            "cmlimit": "max",
        }
        if continuation:
            params["cmcontinue"] = continuation
        payload = client.get_json("https://en.wikipedia.org/w/api.php", params)
        for item in payload.get("query", {}).get("categorymembers", []):
            title = str(item.get("title") or "")
            if title and traversable_category(title):
                output.append(title)
        continuation = str(payload.get("continue", {}).get("cmcontinue") or "")
        if not continuation:
            break
    return output


def _deep_category_query(
    client: WikimediaClient,
    *,
    domain: str,
    root_category: str,
    query_category: str,
    limit: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    offset = 0
    requests = 0
    warnings: List[str] = []
    category_name = query_category.removeprefix("Category:")
    root_title = f"Category:{root_category.removeprefix('Category:')}"
    query_title = f"Category:{category_name}"
    while len(rows) < limit and offset < 10_000:
        payload = client.get_json(
            "https://en.wikipedia.org/w/api.php",
            {
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "generator": "search",
                "gsrsearch": f'deepcat:"{category_name}"',
                "gsrnamespace": 0,
                "gsrlimit": 500,
                "gsroffset": offset,
                "prop": "langlinks|pageprops",
                "lllang": "zh",
                "lllimit": "max",
                "ppprop": "wikibase_item",
            },
        )
        requests += 1
        warning = str((payload.get("warnings", {}).get("search") or {}).get("warnings") or "")
        if warning and warning not in warnings:
            warnings.append(warning)
        pages = payload.get("query", {}).get("pages", [])
        for raw in pages:
            term = str(raw.get("title") or "").strip()
            target = str(((raw.get("langlinks") or [{}])[0]).get("title") or "").strip()
            qid = str((raw.get("pageprops") or {}).get("wikibase_item") or "")
            key = normalized_term(term)
            if not key or not target or not qid.startswith("Q") or not acceptable_title(term):
                continue
            path = [root_title]
            if query_title != root_title:
                path.append(query_title)
            rows.setdefault(
                key,
                {
                    "term": term,
                    "term_key": key,
                    "target_translations": {"zh": target},
                    "source": "wikipedia_deep_category",
                    "wikipedia_pageid": int(raw.get("pageid")),
                    "wikidata_qid": qid,
                    "domain": domain,
                    "domain_root_categories": [root_title],
                    "category_depth": len(path) - 1,
                    "category_path": path,
                    "category_query": f'deepcat:"{category_name}"',
                },
            )
        next_offset = payload.get("continue", {}).get("gsroffset")
        if next_offset is None or not pages:
            break
        offset = int(next_offset)
    return list(rows.values()), {
        "query_category": query_title,
        "requests": requests,
        "rows_with_zh_langlink": len(rows),
        "warnings": warnings,
    }


def collect_deep_category_rows(
    client: WikimediaClient,
    *,
    domain: str,
    root_categories: Sequence[str],
    target_rows: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    query_stats: List[Dict[str, Any]] = []
    queries: List[Tuple[str, str]] = []
    category_queries = WIKIPEDIA_DEEPCAT_CATEGORIES[domain]
    for index, category in enumerate(category_queries):
        root = root_categories[index % len(root_categories)]
        queries.append((root, f"Category:{category.removeprefix('Category:')}"))

    for root, query_category in queries:
        remaining = target_rows - len(rows)
        if remaining <= 0:
            break
        batch, stats = _deep_category_query(
            client,
            domain=domain,
            root_category=root,
            query_category=query_category,
            limit=remaining,
        )
        query_stats.append(stats)
        for row in batch:
            rows.setdefault(normalized_term(str(row["term"])), row)
        print(
            f"[{domain}] deepcat={query_category} rows={len(rows)}/{target_rows}",
            flush=True,
        )

    result = list(rows.values())
    result.sort(
        key=lambda row: (
            int(row.get("category_depth", 99)),
            stable_hash(domain, str(row.get("wikipedia_pageid", ""))),
        )
    )
    return result[:target_rows], {
        "backend": "cirrussearch_deepcat",
        "configured_categories": list(category_queries),
        "queries": query_stats,
        "query_count": len(query_stats),
        "rows_with_zh_langlink": len(result),
    }


def _resolve_wikidata_label_batch(
    client: WikimediaClient,
    pages: Sequence[PageCandidate],
    *,
    domain: str,
    root_categories: Sequence[str],
) -> List[Dict[str, Any]]:
    page_payload = client.get_json(
        "https://en.wikipedia.org/w/api.php",
        {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "pageids": "|".join(str(page.pageid) for page in pages),
            "prop": "langlinks|pageprops",
            "lllang": "zh",
            "lllimit": "max",
            "ppprop": "wikibase_item",
        },
    )
    by_id = {page.pageid: page for page in pages}
    output: List[Dict[str, Any]] = []
    for raw in page_payload.get("query", {}).get("pages", []):
        try:
            pageid = int(raw.get("pageid"))
        except (TypeError, ValueError):
            continue
        page = by_id.get(pageid)
        qid = str((raw.get("pageprops") or {}).get("wikibase_item") or "")
        target = str(((raw.get("langlinks") or [{}])[0]).get("title") or "").strip()
        if page is None or not target or not qid.startswith("Q"):
            continue
        if not acceptable_title(page.title):
            continue
        key = normalized_term(page.title)
        output.append(
            {
                "term": page.title,
                "term_key": key,
                "target_translations": {"zh": target},
                "source": "wikipedia_category",
                "wikipedia_pageid": page.pageid,
                "wikidata_qid": qid,
                "domain": domain,
                "domain_root_categories": [page.category_path[0]],
                "category_depth": page.depth,
                "category_path": list(page.category_path),
            }
        )
    return output


def _chunks(values: Sequence[PageCandidate], size: int) -> Iterable[Sequence[PageCandidate]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def collect_category_rows(
    client: WikimediaClient,
    *,
    domain: str,
    root_categories: Sequence[str],
    target_rows: int,
    max_depth: int,
    max_categories: int,
    max_candidates: int,
    max_members_per_category: int,
    workers: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    queue = deque(
        (f"Category:{root}", 0, (f"Category:{root}",)) for root in root_categories
    )
    visited_categories: set[str] = set()
    page_candidates: MutableMapping[int, PageCandidate] = {}
    resolved_pageids: set[int] = set()
    rows: MutableMapping[str, Dict[str, Any]] = {}

    def resolve_pending(force: bool = False) -> None:
        pending = [
            page for pageid, page in page_candidates.items() if pageid not in resolved_pageids
        ]
        if not force and len(pending) < 2_000:
            return
        for page in pending:
            resolved_pageids.add(page.pageid)
        batches = list(_chunks(pending, 50))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _resolve_wikidata_label_batch,
                    client,
                    batch,
                    domain=domain,
                    root_categories=root_categories,
                )
                for batch in batches
            ]
            for future in concurrent.futures.as_completed(futures):
                for row in future.result():
                    key = normalized_term(str(row["term"]))
                    current = rows.get(key)
                    if current is None or int(row["category_depth"]) < int(current["category_depth"]):
                        rows[key] = row

    while queue and len(visited_categories) < max_categories and len(page_candidates) < max_candidates:
        category_title, depth, path = queue.popleft()
        if category_title in visited_categories:
            continue
        if depth > 0 and not traversable_category(category_title):
            continue
        visited_categories.add(category_title)
        members = _category_members(
            client,
            category_title,
            max_members=max_members_per_category,
        )
        for item in members:
            namespace = int(item.get("ns", -1))
            title = str(item.get("title") or "")
            if namespace == 14 and depth < max_depth and traversable_category(title):
                queue.append((title, depth + 1, path + (title,)))
            elif namespace == 0 and acceptable_title(title):
                try:
                    pageid = int(item["pageid"])
                except (KeyError, TypeError, ValueError):
                    continue
                candidate = PageCandidate(pageid, title, depth, path)
                current = page_candidates.get(pageid)
                if current is None or candidate.depth < current.depth:
                    page_candidates[pageid] = candidate
        resolve_pending()
        if len(visited_categories) % 50 == 0:
            print(
                f"[{domain}] categories={len(visited_categories)} "
                f"candidates={len(page_candidates)} zh={len(rows)}",
                flush=True,
            )
        if len(rows) >= target_rows:
            break

    resolve_pending(force=True)
    result = list(rows.values())
    result.sort(
        key=lambda row: (
            int(row.get("category_depth", 99)),
            stable_hash(domain, str(row.get("wikipedia_pageid", ""))),
        )
    )
    stats = {
        "categories_visited": len(visited_categories),
        "page_candidates": len(page_candidates),
        "pages_with_zh_wikidata_label": len(result),
        "categories_remaining": len(queue),
    }
    return result[:target_rows], stats


def merge_candidates(
    rdf_rows: Sequence[Dict[str, Any]],
    category_rows: Sequence[Dict[str, Any]],
    *,
    domain: str,
    limit: int,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for row in (*rdf_rows, *category_rows):
        key = normalized_term(str(row.get("term") or ""))
        if key and key not in merged:
            merged[key] = dict(row)
    rows = list(merged.values())
    rows.sort(
        key=lambda row: (
            0 if str(row.get("source") or "").startswith("wikidata_") else 1,
            int(row.get("category_depth", 99)),
            stable_hash(domain, str(row.get("term_key") or row.get("term") or "")),
        )
    )
    return rows[:limit]


def parse_domains(raw: str) -> List[str]:
    domains = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [domain for domain in domains if domain not in WIKIDATA_ROOTS]
    if unknown:
        raise ValueError(f"unknown domains: {', '.join(unknown)}")
    return domains


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--domains", default=",".join(WIKIDATA_ROOTS))
    ap.add_argument("--target-rows", type=int, default=13_000)
    ap.add_argument("--rdf-limit", type=int, default=20_000)
    ap.add_argument(
        "--skip-rdf",
        action="store_true",
        help="Skip Wikidata SPARQL roots and collect only category-provenance rows.",
    )
    ap.add_argument("--exact-type-limit", type=int, default=10_000)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--max-categories", type=int, default=4_000)
    ap.add_argument("--max-candidates", type=int, default=100_000)
    ap.add_argument("--max-members-per-category", type=int, default=5_000)
    ap.add_argument("--category-backend", choices=("deepcat", "bfs"), default="deepcat")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--min-request-interval-s", type=float, default=0.2)
    ap.add_argument("--timeout-s", type=float, default=60.0)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument(
        "--user-agent",
        default="AutoTerm-SST/1.0 (academic domain glossary evaluation)",
    )
    args = ap.parse_args()
    if args.target_rows <= 0 or args.rdf_limit <= 0:
        raise SystemExit("--target-rows and --rdf-limit must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    try:
        domains = parse_domains(args.domains)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.out_dir.mkdir(parents=True, exist_ok=True)
    client = WikimediaClient(
        user_agent=args.user_agent,
        timeout_s=args.timeout_s,
        retries=args.retries,
        min_interval_s=args.min_request_interval_s,
        cache_path=args.out_dir / "wikimedia_api_cache.sqlite3",
    )
    report: Dict[str, Any] = {
        "target_rows": args.target_rows,
        "rdf_query": "P31/P279* UNION P279+",
        "domains": {},
    }

    for domain in domains:
        root_qid = WIKIDATA_ROOTS[domain]
        root_categories = WIKIPEDIA_ROOT_CATEGORIES[domain]
        rdf_rows: List[Dict[str, Any]] = []
        exact_p31_rows: List[Dict[str, Any]] = []
        exact_p31_stats: List[Dict[str, Any]] = []
        rdf_error = ""
        query_sha256 = ""
        if args.skip_rdf:
            rdf_error = "skipped by --skip-rdf"
        elif domain in WIKIDATA_SKIP_ABSTRACT_ROOT:
            rdf_error = "skipped: abstract root is too broad for a domain-pure glossary"
        else:
            try:
                rdf_rows, query_sha256 = collect_rdf_rows(
                    client,
                    domain=domain,
                    root_qid=root_qid,
                    limit=args.rdf_limit,
                )
            except (OSError, ValueError, TimeoutError) as exc:
                rdf_error = f"{type(exc).__name__}: {exc}"

        exact_type_qids = WIKIDATA_EXACT_P31_TYPES.get(domain, ())
        if exact_type_qids:
            exact_p31_rows, exact_p31_stats = collect_exact_p31_rows(
                client,
                domain=domain,
                type_qids=exact_type_qids,
                per_type_limit=args.exact_type_limit,
            )
        core_rows = merge_candidates(
            rdf_rows,
            exact_p31_rows,
            domain=domain,
            limit=args.target_rows,
        )

        category_target = max(0, math.ceil((args.target_rows - len(core_rows)) * 1.2))
        category_rows: List[Dict[str, Any]] = []
        category_stats: Dict[str, int] = {}
        if len(core_rows) < args.target_rows:
            if args.category_backend == "deepcat":
                category_rows, category_stats = collect_deep_category_rows(
                    client,
                    domain=domain,
                    root_categories=root_categories,
                    target_rows=category_target,
                )
            else:
                shallow_categories = WIKIPEDIA_DEEPCAT_CATEGORIES[domain]
                category_rows, category_stats = collect_category_rows(
                    client,
                    domain=domain,
                    root_categories=shallow_categories,
                    target_rows=category_target,
                    max_depth=args.max_depth,
                    max_categories=args.max_categories,
                    max_candidates=args.max_candidates,
                    max_members_per_category=args.max_members_per_category,
                    workers=args.workers,
                )
        rows = merge_candidates(
            core_rows,
            category_rows,
            domain=domain,
            limit=args.target_rows,
        )
        if len(rows) < args.target_rows:
            raise SystemExit(
                f"{domain}: only {len(rows)} auditable bilingual candidates; "
                f"increase crawl depth/candidate limits"
            )
        out = args.out_dir / f"{domain}_wikimedia_candidates.json"
        out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        sources: Dict[str, int] = {}
        for row in rows:
            source = str(row.get("source") or "unknown")
            sources[source] = sources.get(source, 0) + 1
        report["domains"][domain] = {
            "root_qid": root_qid,
            "root_categories": list(root_categories),
            "rdf_query_sha256": query_sha256,
            "rdf_rows": len(rdf_rows),
            "rdf_error": rdf_error,
            "exact_p31_types": list(exact_type_qids),
            "exact_p31_rows": len(exact_p31_rows),
            "exact_p31_stats": exact_p31_stats,
            "category_stats": category_stats,
            "output_rows": len(rows),
            "sources": sources,
            "path": str(out),
        }
        print(f"[{domain}] rows={len(rows)} sources={sources} -> {out}", flush=True)

    report_path = args.out_dir / "wikimedia_collection_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
