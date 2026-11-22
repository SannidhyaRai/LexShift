from collections import defaultdict
from pathlib import Path
import argparse
import csv
import json
import re
from urllib.parse import unquote, urlsplit

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from m2_statute.extractor import extract_mentions
from m2_statute.mapping import REPO_ROOT, base_section, load_config, load_map
from m2_statute.neardup import pdf_text

CRAWL_LOG=REPO_ROOT/"data"/"crawl_log.csv"


def context(text,m):
    a=max(0,m.start-250)
    b=min(len(text),m.end+250)
    return " ".join((text[a:m.start]+" "+text[m.end:b]).split())


def scores(a,b):
    x=set(re.findall(r"\w+",a.casefold()))
    y=set(re.findall(r"\w+",b.casefold()))
    jac=len(x&y)/len(x|y) if x|y else 0.0
    try:
        v=TfidfVectorizer()
        tf=v.fit_transform([a,b])
        cos=float(cosine_similarity(tf[0],tf[1])[0,0])
    except ValueError:
        cos=0.0
    return cos,jac


def run(judgments_path=None,map_path=None,out_path=None,cos_threshold=0.8,jac_threshold=0.5):
    paths=load_config()["paths"]
    src=Path(judgments_path) if judgments_path else REPO_ROOT/paths["judgments"]
    mp=Path(map_path) if map_path else REPO_ROOT/paths["statute_map"]
    out=Path(out_path) if out_path else REPO_ROOT/"data"/"processed"/"relation_suggestions.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} does not exist; run `python -m m1_index.ingest unpack` first")
    rows=load_map(str(mp))
    targets=set()
    for r in rows:
        targets.add((r.old_act,base_section(r.old_section)))
        targets.add((r.new_act,base_section(r.new_section)))
    judgments={}
    with src.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec=json.loads(line)
            doc_id=str(rec.get("doc_id", ""))
            if not doc_id or doc_id in judgments:
                continue
            judgments[doc_id]=(rec.get("date") or "2020-01-01",rec.get("text") or "")
    crawled={}
    if CRAWL_LOG.exists():
        with CRAWL_LOG.open(encoding="utf-8",newline="") as f:
            for rec in csv.DictReader(f):
                if rec.get("status")!="200" or str(rec.get("robots_allowed", "")).lower()!="true":
                    continue
                purl=urlsplit(rec.get("url", ""))
                if purl.netloc.lower()!="indian-supreme-court-judgments.s3.amazonaws.com":
                    continue
                url=unquote(purl.path)
                name=url.rsplit("/",1)[-1]
                if not name.lower().endswith(".pdf"):
                    continue
                doc_id=name[:-4]
                if not doc_id or doc_id in crawled:
                    continue
                page=rec.get("page_path") or rec.get("duplicate_of")
                if not page:
                    continue
                p=Path(page)
                if not p.is_absolute():
                    p=REPO_ROOT/p
                if not p.exists():
                    continue
                try:
                    text=pdf_text(p)
                except Exception:
                    continue
                if not text.strip():
                    continue
                date=judgments.get(doc_id, ("2020-01-01", ""))[0]
                if doc_id not in judgments:
                    m=re.search(r"/year=(\d{4})/",url)
                    if m:
                        date=m.group(1)+"-01-01"
                crawled[doc_id]=(date,text)
    texts=defaultdict(set)
    docs=defaultdict(set)
    for doc_id,(j_date,old_text) in judgments.items():
        text=crawled.get(doc_id,(j_date,old_text))[1] or old_text
        for m in extract_mentions(text,j_date):
            key=(m.act,base_section(m.section))
            if key not in targets:
                continue
            s=context(text,m)
            if s:
                texts[key].add(s)
                docs[key].add(doc_id)
    for doc_id,(j_date,text) in crawled.items():
        if doc_id in judgments:
            continue
        for m in extract_mentions(text,j_date):
            key=(m.act,base_section(m.section))
            if key not in targets:
                continue
            s=context(text,m)
            if s:
                texts[key].add(s)
                docs[key].add(doc_id)
    fields=["old_act","old_section","new_act","new_section","map_relation","weight","map_source","old_judgments","new_judgments","cosine","jaccard","suggestion"]
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n")
        w.writeheader()
        for r in rows:
            old=(r.old_act,base_section(r.old_section))
            new=(r.new_act,base_section(r.new_section))
            a=" ".join(sorted(texts[old]))
            b=" ".join(sorted(texts[new]))
            cos,jac=scores(a,b) if a and b else ("","")
            if not a or not b:
                suggestion="insufficient_evidence"
            elif cos>=cos_threshold and jac>=jac_threshold:
                suggestion="equivalent_candidate"
            else:
                suggestion="manual_review"
            w.writerow({"old_act":r.old_act,"old_section":r.old_section,"new_act":r.new_act,"new_section":r.new_section,"map_relation":r.relation,"weight":r.weight,"map_source":r.source,"old_judgments":len(docs[old]),"new_judgments":len(docs[new]),"cosine":round(cos,4) if cos!="" else "","jaccard":round(jac,4) if jac!="" else "","suggestion":suggestion})
    return len(rows),out


def solve():
    ap=argparse.ArgumentParser()
    ap.add_argument("--judgments")
    ap.add_argument("--map")
    ap.add_argument("--out")
    ap.add_argument("--cos-threshold",type=float,default=0.8)
    ap.add_argument("--jaccard-threshold",type=float,default=0.5)
    a=ap.parse_args()
    if not 0<=a.cos_threshold<=1 or not 0<=a.jaccard_threshold<=1:
        ap.error("similarity thresholds must be in [0,1]")
    n,out=run(a.judgments,a.map,a.out,a.cos_threshold,a.jaccard_threshold)
    print(f"wrote {n} relation suggestions to {out}")


if __name__=="__main__":
    solve()
