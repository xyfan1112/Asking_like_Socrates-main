#!/usr/bin/env python3
"""Audit a canonical full raw run without treating verifier rejection as a crash."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows=[]
    with path.open(encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            if not line.strip():
                continue
            try: rows.append(json.loads(line))
            except Exception as exc: rows.append({'_bad_json_line':i,'_error':str(exc)})
    return rows


def coordinate_mutated(original: str, rewritten: str) -> bool:
    a=re.sub(r'\s+','',original or '').lower(); b=re.sub(r'\s+','',rewritten or '').lower()
    return ('[0,1000]' in a and '[0,1000]' not in b) or ('[0,100]' in a and '[0,100]' not in b)


def failure_kind(row: dict[str, Any]) -> str:
    loop=row.get('loop_result') if isinstance(row.get('loop_result'),dict) else {}
    if loop.get('success') is True:
        return 'success'
    if loop.get('error'):
        return str(loop['error'])
    history=loop.get('chat_history') or []
    if any(turn.get('V_decision')=='REJECT' for turn in history):
        return 'verifier_reject'
    if loop.get('final_answer'):
        return 'final_not_accepted'
    return 'unknown_failure'


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw',type=Path,required=True)
    ap.add_argument('--api-log',type=Path)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    rows=read_jsonl(args.raw)
    tasks=Counter(); outcomes=Counter(); finals=Counter(); mutations=0; bad=0
    for row in rows:
        if '_bad_json_line' in row:
            bad+=1; continue
        task=str(row.get('task','unknown')); tasks[task]+=1
        outcome=failure_kind(row); outcomes[(task,outcome)]+=1
        loop=row.get('loop_result') or {}
        if loop.get('final_answer'): finals[task]+=1
        if coordinate_mutated(str(row.get('query','')),str(row.get('rewritten_query',''))): mutations+=1
    api={
        'calls':0,
        'length_truncation':0,
        'api_errors':0,
        'format_repairs':0,
        'unrepaired':0,
        'unrepaired_fail_closed':0,
        'unrepaired_unsafe':0,
        'reasoner_format_repairs':0,
        'reasoner_format_unrepaired':0,
        'reasoner_format_unrepaired_fail_closed':0,
        'perceiver_response_repairs':0,
        'perceiver_response_unrepaired':0,
        'perceiver_response_unrepaired_fail_closed':0,
        'perceiver_calls':0,
        'perceiver_context_missing':0,
        'perceiver_focus_roi_missing':0,
        'classification_perceiver_calls':0,
        'classification_focus_roi_missing':0,
    }
    if args.api_log and args.api_log.exists():
        for row in read_jsonl(args.api_log):
            event=row.get('event')
            if event=='api_call':
                api['calls']+=1
                if str(row.get('finish_reason','')).lower()=='length': api['length_truncation']+=1
                if row.get('role')=='perceiver':
                    api['perceiver_calls']+=1
                    if not row.get('original_context_attached'):
                        api['perceiver_context_missing']+=1
                    if not row.get('focus_roi_attached'):
                        api['perceiver_focus_roi_missing']+=1
                    if row.get('original_context_task')=='ref_classification':
                        api['classification_perceiver_calls']+=1
                        if not row.get('focus_roi_attached'):
                            api['classification_focus_roi_missing']+=1
            elif event=='api_error': api['api_errors']+=1
            elif event=='perceiver_context_missing': api['perceiver_context_missing']+=1
            elif event=='reasoner_format_repaired':
                api['format_repairs']+=1
                api['reasoner_format_repairs']+=1
            elif event=='reasoner_format_unrepaired':
                api['unrepaired']+=1
                api['reasoner_format_unrepaired']+=1
                if bool(row.get('fail_closed')):
                    api['unrepaired_fail_closed']+=1
                    api['reasoner_format_unrepaired_fail_closed']+=1
                else:
                    api['unrepaired_unsafe']+=1
            elif event=='perceiver_response_repaired':
                api['format_repairs']+=1
                api['perceiver_response_repairs']+=1
            elif event=='perceiver_response_unrepaired':
                api['unrepaired']+=1
                api['perceiver_response_unrepaired']+=1
                if bool(row.get('fail_closed')):
                    api['unrepaired_fail_closed']+=1
                    api['perceiver_response_unrepaired_fail_closed']+=1
                else:
                    api['unrepaired_unsafe']+=1
    by_task={}
    for task,n in tasks.items():
        row={'input':n,'final_present':finals[task]}
        for (t,o),count in outcomes.items():
            if t==task: row[o]=count
        by_task[task]=row
    api_log_missing=not args.api_log or not args.api_log.exists()
    # A response with finish_reason='length' is never returned directly by
    # the adapter: it is repaired or replaced by a fail-closed fallback. Treat
    # recovered truncations as diagnostics. A leaked response remains a hard
    # failure through unrepaired_unsafe > 0.
    warnings=[]
    if api['length_truncation']:
        warnings.append(f"recovered_length_truncation={api['length_truncation']}")
    if api['unrepaired_fail_closed']:
        warnings.append(f"unrepaired_fail_closed={api['unrepaired_fail_closed']}")
    hard=(
        bad
        or mutations
        or api_log_missing
        or (sum(tasks.values()) and not api['perceiver_calls'])
        or api['api_errors']
        or api['unrepaired_unsafe']
        or api['perceiver_context_missing']
        or api['perceiver_focus_roi_missing']
        or api['classification_focus_roi_missing']
    )
    report={
        'status':'FAIL' if hard else ('PASS_WITH_WARNINGS' if warnings else 'PASS_WITH_MODEL_REJECTIONS'),
        'raw':str(args.raw),'rows':sum(tasks.values()),'tasks':by_task,
        'coordinate_rewrite_mutations':mutations,'bad_json_lines':bad,'api':api,
        'api_log_missing':api_log_missing,
        'warnings':warnings,
        'hard_failure':bool(hard),
    }
    out=args.output or args.raw.with_suffix('.full_audit.json')
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"[OFFICIAL FULL AUDIT] {report['status']}")
    print(f"  rows={report['rows']} tasks={by_task}")
    print(f"  coordinate_mutations={mutations} bad_json={bad}")
    print(f"  api={api}")
    if warnings: print(f"  warnings={warnings}")
    print(f"  report: {out}")
    return 2 if hard else 0

if __name__=='__main__':
    raise SystemExit(main())
