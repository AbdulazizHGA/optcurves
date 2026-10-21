"""Rerun the API part of the literature check and store the responses, dated, so that a later check can be compared.

    python3 scripts/literature_check.py            # writes docs/literature-runs/<UTC date>/

Standard library only. Stored per query: the URL, the UTC time, the HTTP status and the records with identifiers,
titles, authors and dates. Abstracts are left out on purpose (they are the authors' text, and the identifiers are
enough to look a record up again). Google Scholar has no API and is not queried here; see the dated logs in docs/.
"""
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

PAPER_ARXIV = "2503.10138"            # Barzilai, Shamir, Zamani, Are Convex Optimization Curves Convex?
OPENREVIEW_FORUM = "TZtpxselK2"       # its TMLR forum
S2_SEARCHES = ["optimization curve convex gradient descent",
               "optimization curve concave consecutive steps gradient descent",
               "worst-case convexity violation gradient descent function values",
               "convexity optimization curve strongly convex",
               "convex optimization curves step size 1.75"]
# The 25 arXiv queries of docs/literature-check-2026-10-20.md, in the same order.
ARXIV_QUERIES = [
    'all:"optimization curve"', 'all:"optimization curves"',
    'abs:"optimization curve" AND abs:"strongly convex"', 'abs:"optimization curve" AND abs:"worst-case"',
    'abs:"optimization curve" AND abs:"gradient descent"', 'all:"convex optimization curves"',
    'abs:"convexity" AND abs:"gradient descent" AND abs:"1.75"', 'all:"1.75/L"',
    'abs:"curve" AND abs:"convex" AND abs:"gradient descent" AND abs:"step size" AND abs:"concave"',
    'abs:"second difference" AND abs:"gradient descent"', 'abs:"discrete convexity" AND abs:"gradient"',
    'abs:"function values" AND abs:"convex sequence" AND abs:"gradient"',
    'abs:"performance estimation" AND abs:"strongly convex" AND abs:"gradient descent" AND abs:"step size"',
    'abs:"worst-case" AND abs:"strongly convex" AND abs:"gradient descent" AND abs:"convexity" AND abs:"curve"',
    'abs:"Barzilai" AND abs:"Shamir"',
    'abs:"decrease" AND abs:"monotone" AND abs:"gradient descent" AND abs:"function value" AND abs:"differences"',
    'ti:"optimization curve"', 'abs:"loss curve" AND abs:"convex" AND abs:"gradient descent" AND abs:"step size"',
    'abs:"convexity of the loss curve"',
    'abs:"performance estimation" AND abs:"function values" AND abs:"convexity" AND abs:"gradient descent"',
    'abs:"sequence of function values" AND abs:"convex" AND abs:"gradient descent" AND abs:"smooth"',
    'abs:"objective values" AND abs:"convex sequence"',
    'abs:"consecutive" AND abs:"decrease" AND abs:"gradient descent" AND abs:"smooth convex" AND abs:"step size"',
    'abs:"optimization curves" AND abs:"heavy ball"', 'abs:"optimization curve" AND abs:"proximal"',
]
AGENT = "optcurves-literature-check/1.0 (https://github.com/AbdulazizHGA/optcurves)"


def fetch(url, tries=6, pause=3.0):
    """(HTTP status, body text or None, UTC time). Retries on 429 and 5xx with growing pauses."""
    for attempt in range(tries):
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": AGENT}), timeout=60) as r:
                return r.status, r.read().decode("utf-8"), stamp
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500 or attempt == tries - 1:
                return e.code, None, stamp
        except urllib.error.URLError as e:
            if attempt == tries - 1:
                return f"error: {e.reason}", None, stamp
        time.sleep(pause * 2 ** attempt)


def semantic_scholar():
    out = []
    base = "https://api.semanticscholar.org/graph/v1/paper"
    for label, url in [("paper", f"{base}/arXiv:{PAPER_ARXIV}?fields=title,year,citationCount,externalIds"),
                       ("citations", f"{base}/arXiv:{PAPER_ARXIV}/citations?fields=title,year,externalIds,authors"
                                     "&limit=100")]:
        status, body, stamp = fetch(url)
        data = json.loads(body) if body else None
        if label == "citations" and data:
            data = [{"title": c["citingPaper"].get("title"), "year": c["citingPaper"].get("year"),
                     "externalIds": c["citingPaper"].get("externalIds"),
                     "authors": [a.get("name") for a in c["citingPaper"].get("authors") or []]}
                    for c in data.get("data", [])]
        out.append({"query": label, "url": url, "retrieved_utc": stamp, "http_status": status, "result": data})
        time.sleep(3)
    for q in S2_SEARCHES:
        url = ("https://api.semanticscholar.org/graph/v1/paper/search?"
               + urllib.parse.urlencode({"query": q, "limit": 20, "fields": "title,year,externalIds"}))
        status, body, stamp = fetch(url)
        data = json.loads(body) if body else None
        if data:
            data = {"total": data.get("total"), "records": data.get("data", [])}
        out.append({"query": q, "url": url, "retrieved_utc": stamp, "http_status": status, "result": data})
        time.sleep(3)
    return out


def openalex():
    out = []
    url = f"https://api.openalex.org/works/doi:10.48550/arXiv.{PAPER_ARXIV}"
    status, body, stamp = fetch(url)
    work = json.loads(body) if body else {}
    out.append({"query": "work", "url": url, "retrieved_utc": stamp, "http_status": status,
                "result": {k: work.get(k) for k in ("id", "doi", "display_name", "publication_year",
                                                     "cited_by_count")} if work else None})
    if work.get("id"):
        wid = work["id"].rsplit("/", 1)[-1]
        url = (f"https://api.openalex.org/works?filter=cites:{wid}&per-page=100"
               "&select=id,doi,display_name,publication_year")
        status, body, stamp = fetch(url)
        data = json.loads(body) if body else None
        out.append({"query": f"cites:{wid}", "url": url, "retrieved_utc": stamp, "http_status": status,
                    "result": {"count": data["meta"]["count"], "records": data["results"]} if data else None})
    return out


def arxiv():
    out = []
    ns = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/"}
    for q in ARXIV_QUERIES:
        url = ("https://export.arxiv.org/api/query?"
               + urllib.parse.urlencode({"search_query": q, "start": 0, "max_results": 60,
                                         "sortBy": "submittedDate", "sortOrder": "descending"}))
        status, body, stamp = fetch(url)
        result = None
        if body:
            root = ET.fromstring(body)
            result = {"total": int(root.findtext("o:totalResults", "0", ns)),
                      "records": [{"id": e.findtext("a:id", "", ns).rsplit("/abs/", 1)[-1],
                                   "title": " ".join(e.findtext("a:title", "", ns).split()),
                                   "published": e.findtext("a:published", "", ns),
                                   "authors": [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)]}
                                  for e in root.findall("a:entry", ns)]}
        out.append({"query": q, "url": url, "retrieved_utc": stamp, "http_status": status, "result": result})
        time.sleep(3)  # arXiv asks for 3 s between calls
    return out


def openreview():
    """Only whether the forum's notes can be read through the API, and their count and kinds: the reviews are read
    by a person, not stored."""
    url = f"https://api2.openreview.net/notes?forum={OPENREVIEW_FORUM}"
    status, body, stamp = fetch(url, tries=2)
    result = None
    if body:
        try:
            notes = json.loads(body).get("notes", [])
            result = {"notes": len(notes),
                      "invitations": sorted({inv.rsplit("/", 1)[-1] for n in notes for inv in n.get("invitations", [])})}
        except json.JSONDecodeError:
            result = "not JSON (a browser check page?)"
    return [{"query": f"forum {OPENREVIEW_FORUM}", "url": url, "retrieved_utc": stamp, "http_status": status,
             "result": result}]


def main():
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = Path(__file__).resolve().parents[1] / "docs" / "literature-runs" / day
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for name, run in [("semantic_scholar", semantic_scholar), ("openalex", openalex), ("arxiv", arxiv),
                      ("openreview", openreview)]:
        records = run()
        (out / f"{name}.json").write_text(json.dumps(records, indent=1, ensure_ascii=False) + "\n")
        for r in records:
            index.append({"source": name, "query": r["query"], "http_status": r["http_status"],
                          "retrieved_utc": r["retrieved_utc"]})
        print(f"{name}: {len(records)} queries, statuses {sorted({str(r['http_status']) for r in records})}",
              file=sys.stderr, flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
