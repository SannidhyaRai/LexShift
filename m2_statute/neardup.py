from collections import Counter,defaultdict
from hashlib import blake2b
from html.parser import HTMLParser
from pathlib import Path
import argparse
import heapq
import json
import re

ROOT=Path(__file__).resolve().parents[1]
JUDGMENTS=ROOT/"data"/"processed"/"judgments.jsonl"
PAGE_DIR=ROOT/"data"/"raw"/"crawl"/"pages"
STATUTE_TEXT=ROOT/"data"/"statute_text.jsonl"
OUT=ROOT/"data"/"processed"/"neardup.jsonl"


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text=[]
        self.skip=[]

    def handle_starttag(self,tag,attrs):
        if tag in ("script","style","noscript"):
            self.skip.append(tag)

    def handle_endtag(self,tag):
        if tag in self.skip:
            self.skip.remove(tag)

    def handle_data(self,data):
        if not self.skip and data.strip():
            self.text.append(data.strip())


def pdf_text(path):
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf
    with pymupdf.open(path) as doc:
        return " ".join(page.get_text("text") for page in doc)


def signature(text,k,num_perm):
    words=re.findall(r"\w+",text.casefold())
    vals=set()
    for i in range(len(words)-k+1):
        s=" ".join(words[i:i+k]).encode("utf-8")
        vals.add(int.from_bytes(blake2b(s,digest_size=8).digest(),"big"))
    heap=[]
    for x in vals:
        if len(heap)<num_perm:
            heapq.heappush(heap,-x)
        elif x < -heap[0]:
            heapq.heapreplace(heap,-x)
    return sorted(-x for x in heap)


def estimate(a,b,num_perm):
    common=set(a)&set(b)
    union=sorted(set(a)|set(b))[:num_perm]
    if not union:
        return 0.0
    return sum(x in common for x in union)/len(union)


def near_duplicates(texts,threshold=0.9,k=5,num_perm=128):
    if not 0<=threshold<=1 or k<1 or num_perm<1:
        raise ValueError("threshold must be in [0,1], k and num_perm must be positive")
    sigs={}
    for doc_id,text in texts.items():
        sig=signature(text,k,num_perm)
        if sig:
            sigs[doc_id]=sig
    posts=defaultdict(list)
    for doc_id,sig in sigs.items():
        for x in sig:
            posts[x].append(doc_id)
    pairs=Counter()
    for ids in posts.values():
        for i in range(len(ids)):
            for j in range(i+1,len(ids)):
                a,b=ids[i],ids[j]
                pairs[(a,b)]+=1
    out=[]
    for (a,b),shared in pairs.items():
        need=max(1,min(len(sigs[a]),len(sigs[b]))//10)
        if shared<need:
            continue
        score=estimate(sigs[a],sigs[b],num_perm)
        if score>=threshold:
            out.append((a,b,score))
    return sorted(out,key=lambda x:(-x[2],x[0],x[1]))


def load_docs(judgments,statutes,pages):
    docs={}
    with judgments.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec=json.loads(line)
            doc_id=rec.get("doc_id")
            text=rec.get("text") or ""
            if doc_id and text:
                docs["judgment:"+str(doc_id)]=text
    if statutes.exists():
        with statutes.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec=json.loads(line)
                act=rec.get("act")
                section=rec.get("section")
                text=rec.get("text") or ""
                if act and section and text:
                    docs["statute:"+str(act)+":"+str(section)]=text
                    
    if pages.exists():
        for path in sorted(pages.iterdir()):
            ext=path.suffix.lower()
            if ext in (".html",".htm"):
                p=TextParser()
                p.feed(path.read_text(encoding="utf-8",errors="replace"))
                text=" ".join(p.text)
            elif ext==".pdf":
                text=pdf_text(path)
            elif ext==".txt":
                text=path.read_text(encoding="utf-8",errors="replace")
            else:
                continue
            if text.strip():
                docs["crawl:"+path.stem]=text
    return docs


def solve():
    ap=argparse.ArgumentParser()
    ap.add_argument("--judgments",default=str(JUDGMENTS))
    ap.add_argument("--statute-text",default=str(STATUTE_TEXT))
    ap.add_argument("--pages",default=str(PAGE_DIR))
    ap.add_argument("--out",default=str(OUT))
    ap.add_argument("--threshold",type=float,default=0.9)
    ap.add_argument("--k",type=int,default=5)
    ap.add_argument("--num-perm",type=int,default=128)
    a=ap.parse_args()
    docs=load_docs(Path(a.judgments),Path(a.statute_text),Path(a.pages))
    pairs=near_duplicates(docs,a.threshold,a.k,a.num_perm)
    out=Path(a.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf-8",newline="\n") as f:
        for doc_a,doc_b,score in pairs:
            f.write(json.dumps({"doc_a":doc_a,"doc_b":doc_b,"jaccard_estimate":round(score,4)})+"\n")
    print(f"loaded {len(docs)} documents; wrote {len(pairs)} pairs to {out}")


if __name__=="__main__":
    solve()
