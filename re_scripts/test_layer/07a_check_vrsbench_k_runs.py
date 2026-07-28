#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--jsonl',required=True); ap.add_argument('--k',type=int,default=5); args=ap.parse_args()
    counts=Counter(); duplicate=0; seen=set(); bad=0
    with Path(args.jsonl).open(encoding='utf-8') as f:
        for n,line in enumerate(f,1):
            if not line.strip(): continue
            try:r=json.loads(line)
            except Exception: bad+=1; continue
            q=r.get('question_id',r.get('id')); sid=r.get('sample_id',r.get('run_id',0)); key=(q,sid)
            if key in seen: duplicate+=1
            seen.add(key); counts[q]+=1
    dist=Counter(counts.values()); incomplete=sum(1 for v in counts.values() if v!=args.k)
    print(json.dumps({'unique_questions':len(counts),'unique_question_sample_pairs':len(seen),'duplicate_pairs':duplicate,'bad_json_lines':bad,'runs_per_question_distribution':dict(dist),'questions_not_equal_k':incomplete,'expected_k':args.k,'status':'PASS' if not duplicate and not bad and not incomplete else 'FAIL'},ensure_ascii=False,indent=2))
    return 0 if not duplicate and not bad and not incomplete else 2
if __name__=='__main__': raise SystemExit(main())
