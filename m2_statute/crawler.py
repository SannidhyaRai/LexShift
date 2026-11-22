#Imports
from collections import defaultdict, deque
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen
from xml.etree import ElementTree
import argparse
import csv
import hashlib
import json
import time
from urllib.robotparser import RobotFileParser

ROOT=Path(__file__).resolve().parents[1] #Sets repo root to parent of m2_statute
PAGE_DIR=ROOT/"data"/"raw"/"crawl"/"pages" #Sets folder for page bodies
STATE=ROOT/"data"/"raw"/"crawl"/"state.json" #Sets path for saved pending URL's
LOG=ROOT/"data"/"crawl_log.csv" #Sets crawl log paths
AGENT="LexShiftCrawler/0.1"
FIELDS=["url","depth","status","robots_status","robots_allowed","bytes","fetched_at","sha256","duplicate_of","page_path","delay_s","error"]
#Defines the columns written to log

#Based on HTMLParser
class PageParser(HTMLParser):
    #Initialize
    def __init__(self):
        super().__init__(convert_charrefs=True) #allows conversion of HTML References
        self.links=[]
        self.skip=[]

    def handle_starttag(self,tag,attrs):
        if tag in ("script","style","noscript"):  
            self.skip.append(tag) #Adds to ignore stack
        if tag=="a":
            for k,v in attrs:
                if k=="href" and v:
                    self.links.append(v)

    def handle_endtag(self,tag):
        if tag in self.skip:
            self.skip.remove(tag)

class NoRedirect(HTTPRedirectHandler):  #Redirect Handler
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        return None

#URL Validation and Cleaning
def norm(u,base=None):
    if base:
        u=urljoin(base,u)  #Resolves relative link if base is provided
    p=urlsplit(u.strip())
    if p.scheme.lower() not in ("http","https") or not p.hostname:
        return None
    scheme=p.scheme.lower() #Lowercases schema and hostname
    host=p.hostname.lower()
    try:
        port=p.port
    except ValueError:
        return None  #Invalid URL's return NULL
    if port and not (scheme=="http" and port==80 or scheme=="https" and port==443):
        host=f"{host}:{port}"
    netloc=host
    if p.username or p.password:
        return None
    path=p.path or "/"
    return urlunsplit((scheme,netloc,path,p.query,""))


def origin(u):  #Returns the scheme and host, used to find robots.txt
    p=urlsplit(u)
    return f"{p.scheme}://{p.netloc}"


def host_key(u):  #Returns hostname and port
    p=urlsplit(u)
    return p.hostname.lower()+(f":{p.port}" if p.port else "")


def robots(u,agent):
    ru=origin(u)+"/robots.txt"
    rp=RobotFileParser(ru)
    try:
        req=Request(ru,headers={"User-Agent":agent})
        with urlopen(req,timeout=15) as r:
            status=r.status
            data=r.read(512000).decode("utf-8","replace")
        rp.parse(data.splitlines())
        return rp,status
    except HTTPError as e:
        status=e.code
        if status in (404,410):
            rp.parse([]) #Treated as no rules
        else:
            rp.parse(["User-agent: *","Disallow: /"])  #Network faliures treated as disallow
        return rp,status
    except (URLError,TimeoutError,OSError):
        rp.parse(["User-agent: *","Disallow: /"])
        return rp,"error"
#We still need to check the terms of service on automated agents.

def wait_host(h,last,delay):  #last sores time of previous request, calculates delay
    t=max(0.0,last.get(h,0.0)+delay-time.monotonic())
    if t:
        time.sleep(t)
    return round(t,3)


def load_state(resume):  #Restores pending URL/depth pairs
    if resume and STATE.exists():
        try:
            s=json.loads(STATE.read_text(encoding="utf-8"))
            return deque((u,d) for u,d in s.get("queue",[]))
        except (OSError,ValueError,TypeError):
            pass
    return deque()


def save_state(q):  #Creates the state directory, writes pending queue to a JSON file
    STATE.parent.mkdir(parents=True,exist_ok=True)
    tmp=STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"queue":list(q)},ensure_ascii=False),encoding="utf-8")
    tmp.replace(STATE)


def crawl_s3(prefix="data/pdf/year=2025/",max_files=10,delay_s=1.0,user_agent=AGENT,max_bytes=5000000,list_page_size=1000):
    bucket="indian-supreme-court-judgments"
    base=f"https://{bucket}.s3.amazonaws.com/"
    if not prefix or max_files<1 or not 1<=list_page_size<=1000 or delay_s<0 or max_bytes<1:
        raise ValueError("invalid S3 crawl settings")
    prefix=prefix.lstrip("/")
    PAGE_DIR.mkdir(parents=True,exist_ok=True)
    LOG.parent.mkdir(parents=True,exist_ok=True)
    exists=LOG.exists() and LOG.stat().st_size>0
    def log_row(rec):
        nonlocal exists
        with LOG.open("a",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=FIELDS)
            if not exists:
                w.writeheader()
                exists=True
            w.writerow(rec)
        rows.append(rec)

    seen=set()
    hashes={}
    if LOG.exists():
        with LOG.open(encoding="utf-8",newline="") as f:
            for r in csv.DictReader(f):
                if r.get("url"):
                    seen.add(r["url"])
                if r.get("sha256") and r.get("page_path"):
                    hashes.setdefault(r["sha256"],r["page_path"])
    rp,status=robots(base,user_agent)
    rows=[]
    last=time.monotonic()
    token=""
    keys=[]
    opener=build_opener(NoRedirect())
    while len(keys)<max_files:
        params={"list-type":"2","prefix":prefix,"max-keys":str(list_page_size)}
        if token:
            params["continuation-token"]=token
        u=base+"?"+urlencode(params)
        rec={k:"" for k in FIELDS}
        rec.update({"url":u,"depth":0,"fetched_at":datetime.now(timezone.utc).isoformat(timespec="seconds")})
        rec["robots_status"]=status
        rec["robots_allowed"]=rp.can_fetch(user_agent,u)
        if not rec["robots_allowed"]:
            rec["status"]="robots_blocked"
            log_row(rec)
            break
        rec["delay_s"]=wait_host(host_key(u),{host_key(u):last},delay_s)
        try:
            req=Request(u,headers={"User-Agent":user_agent})
            with opener.open(req,timeout=20) as r:
                data=r.read(max_bytes+1)
                rec["status"]=r.status
            last=time.monotonic()
            rec["bytes"]=len(data)
            if len(data)>max_bytes:
                rec["status"]="too_large"
                log_row(rec)
                break
            rec["sha256"]=hashlib.sha256(data).hexdigest()
            root=ElementTree.fromstring(data)
            vals={x.tag.split("}")[-1]:x.text or "" for x in root.iter()}
            for item in root.iter():
                if item.tag.split("}")[-1]!="Contents":
                    continue
                fields={x.tag.split("}")[-1]:x.text or "" for x in item}
                key=fields.get("Key","")
                obj=base+"/".join(quote(x,safe="") for x in key.split("/"))
                if key.lower().endswith(".pdf") and key.startswith(prefix) and obj not in seen:
                    keys.append((obj,key))
                    if len(keys)>=max_files:
                        break
            log_row(rec)
            if vals.get("IsTruncated")!="true" or len(keys)>=max_files:
                break
            token=vals.get("NextContinuationToken","")
            if not token:
                break
        except (HTTPError,URLError,TimeoutError,OSError,ElementTree.ParseError) as e:
            last=time.monotonic()
            rec["status"]=e.code if isinstance(e,HTTPError) else "error"
            rec["error"]=str(e)
            log_row(rec)
            break

    for u,key in keys:
        h=host_key(u)
        rec={k:"" for k in FIELDS}
        rec.update({"url":u,"depth":1,"fetched_at":datetime.now(timezone.utc).isoformat(timespec="seconds")})
        rec["robots_status"]=status
        rec["robots_allowed"]=rp.can_fetch(user_agent,u)
        if not rec["robots_allowed"]:
            rec["status"]="robots_blocked"
        else:
            rec["delay_s"]=wait_host(h,{h:last},delay_s)
            try:
                req=Request(u,headers={"User-Agent":user_agent})
                with opener.open(req,timeout=20) as r:
                    data=r.read(max_bytes+1)
                    rec["status"]=r.status
                last=time.monotonic()
                rec["bytes"]=len(data)
                if len(data)>max_bytes:
                    rec["status"]="too_large"
                else:
                    sha=hashlib.sha256(data).hexdigest()
                    rec["sha256"]=sha
                    if sha in hashes:
                        rec["duplicate_of"]=hashes[sha]
                    else:
                        path=PAGE_DIR/(sha+".pdf")
                        path.write_bytes(data)
                        rec["page_path"]=str(path.relative_to(ROOT)).replace("\\","/")
                        hashes[sha]=rec["page_path"]
            except (HTTPError,URLError,TimeoutError,OSError) as e:
                last=time.monotonic()
                rec["status"]=e.code if isinstance(e,HTTPError) else "error"
                rec["error"]=str(e)
        log_row(rec)

    return rows

#Main crawl function
def crawl(start_urls,max_pages=50,max_depth=2,delay_s=1.0,resume=True,user_agent=AGENT,max_bytes=5000000):
    seed=[]
    for u in start_urls:
        u=norm(u) #normalizes
        if u and u not in seed:
            seed.append(u)
    if not seed or max_pages<1 or max_depth<0 or delay_s<0 or max_bytes<1:
        raise ValueError("invalid crawl settings or no valid seed URLs") #Reject invalid

    PAGE_DIR.mkdir(parents=True,exist_ok=True)
    LOG.parent.mkdir(parents=True,exist_ok=True)
    q=load_state(resume)
    allowed_hosts={host_key(u) for u in seed}
    queue=defaultdict(deque) #Keeps one queue per origin
    turn=deque() #Provides Round Robin Scheduling
    active=set() #Prevents adding same origin multiple time
    seen=set() #Prevents URL from being queued repeatedly

    def add(u,d):
        u=norm(u)
        if not u or host_key(u) not in allowed_hosts or u in seen or d>max_depth:
            return
        seen.add(u)
        h=origin(u)
        queue[h].append((u,d))
        if h not in active:
            active.add(h)
            turn.append(h)

    rp_cache={}
    rp_status={} #Holds robot results
    host_delay={} #Stores delay for each origin
    last={} #Stores request completion times
    hashes={} #Maps saved content hases to paths
    rows=[] #Collects records
    if LOG.exists(): #Avoid refetching if log already exists
        with LOG.open(encoding="utf-8",newline="") as f:
            for r in csv.DictReader(f):
                if r.get("url"):
                    seen.add(r["url"])
                if r.get("sha256") and r.get("page_path"):
                    hashes.setdefault(r["sha256"],r["page_path"])
    for u,d in q:
        add(u,d)
    for u in seed:
        add(u,0)
    exists=LOG.exists() and LOG.stat().st_size>0
    opener=build_opener(NoRedirect()) #disables automatic redirects
    count=0

    try:
        #Selects one URL at a time, rotates among origins.
        while turn and count<max_pages:
            h=turn.popleft() #Take the next origin
            active.discard(h)

            if not queue[h]: #Skips if no URL's queued
                continue
            u,d=queue[h].popleft()

            if queue[h] and h not in active: 
                turn.append(h)
                active.add(h)
            count+=1

            t=datetime.now(timezone.utc).isoformat(timespec="seconds") #Records the current UTC timestamp
            rec={k:"" for k in FIELDS} #Creates a log record with every field
            rec.update({"url":u,"depth":d,"fetched_at":t,"delay_s":0}) #Fills in the URL, depth, timestamp, and delay

            if h not in rp_cache: #Checks whether robot rules have been fetched
                rp_cache[h],rp_status[h]=robots(u,user_agent) #Fetches and parses origin's robots.txt
                host_delay[h]=max(delay_s,rp_cache[h].crawl_delay(user_agent) or 0)
                last[h]=time.monotonic() #records when robots request finished
            rec["robots_status"]=rp_status[h] #records http status
            rec["robots_allowed"]=rp_cache[h].can_fetch(user_agent,u) #Check whether robots rules allow the particular URL

            if not rec["robots_allowed"]:
                rec["status"]="robots_blocked"

            else:
                rec["delay_s"]=wait_host(h,last,host_delay[h])

                try:
                    req=Request(u,headers={"User-Agent":user_agent})
                    with opener.open(req,timeout=20) as r:
                        status=r.status
                        data=r.read(max_bytes+1)
                        headers=r.headers

                    last[h]=time.monotonic()
                    rec["status"]=status
                    rec["bytes"]=len(data)

                    if len(data)>max_bytes:
                        rec["status"]="too_large"

                    else:
                        sha=hashlib.sha256(data).hexdigest()
                        rec["sha256"]=sha

                        if sha in hashes:
                            rec["duplicate_of"]=hashes[sha]

                        else:
                            ct=headers.get_content_type()
                            ext=".html" if ct=="text/html" else ".pdf" if ct=="application/pdf" else ".bin"
                            path=PAGE_DIR/(sha+ext)
                            path.write_bytes(data)
                            rec["page_path"]=str(path.relative_to(ROOT)).replace("\\","/")
                            hashes[sha]=rec["page_path"]

                            if ct=="text/html" and d<max_depth:
                                p=PageParser()
                                enc=headers.get_content_charset() or "utf-8"
                                try:
                                    p.feed(data.decode(enc,"replace"))
                                except LookupError:
                                    p.feed(data.decode("utf-8","replace"))
                                for link in p.links:
                                    add(link,d+1)

                except HTTPError as e:
                    last[h]=time.monotonic()
                    rec["status"]=e.code
                    loc=e.headers.get("Location")
                    v=norm(loc,u) if loc else None
                    if e.code in (301,302,303,307,308) and v and host_key(v) in allowed_hosts:
                        add(v,d)
                        
                except (URLError,TimeoutError,OSError) as e:
                    last[h]=time.monotonic()
                    rec["status"]="error"
                    rec["error"]=str(e)
            rows.append(rec)

            with LOG.open("a",encoding="utf-8",newline="") as f:
                w=csv.DictWriter(f,fieldnames=FIELDS)
                if not exists:
                    w.writeheader()
                    exists=True
                w.writerow(rec)

            rest=[]

            for x in turn:
                rest.extend(queue[x])

            for x in queue:
                if x not in active:
                    rest.extend(queue[x])

            save_state(deque(rest))

    except KeyboardInterrupt:
        pass

    rest=[]
    for x in turn:
        rest.extend(queue[x])
    for x in queue:
        if x not in active:
            rest.extend(queue[x])
    save_state(deque(rest))
    return rows


def solve():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed")
    ap.add_argument("--s3-prefix")
    ap.add_argument("--max-files",type=int,default=10)
    ap.add_argument("--list-page-size",type=int,default=1000)
    ap.add_argument("--max-pages",type=int,default=50)
    ap.add_argument("--max-depth",type=int,default=2)
    ap.add_argument("--delay",type=float,default=1.0)
    ap.add_argument("--max-bytes",type=int,default=5000000)
    ap.add_argument("--user-agent",default=AGENT)
    ap.add_argument("--no-resume",action="store_true")
    a=ap.parse_args()
    if bool(a.seed)==bool(a.s3_prefix):
        ap.error("provide exactly one of --seed or --s3-prefix")
    if a.s3_prefix:
        rows=crawl_s3(a.s3_prefix,a.max_files,a.delay,a.user_agent,a.max_bytes,a.list_page_size)
    else:
        with open(a.seed,encoding="utf-8") as f:
            seed=[x.strip() for x in f if x.strip() and not x.lstrip().startswith("#")]
        rows=crawl(seed,a.max_pages,a.max_depth,a.delay,not a.no_resume,a.user_agent,a.max_bytes)
    print(f"logged {len(rows)} URLs to {LOG.relative_to(ROOT)}")


if __name__=="__main__":
    solve()
