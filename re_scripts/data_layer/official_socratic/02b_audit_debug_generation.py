#!/usr/bin/env python3
"""Audit debug output: workflow integrity, token truncation, and strict candidates."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'main_layer'))
from main_layer.common import load_settings, read_jsonl as common_read_jsonl, settings_from_cli  # noqa: E402
from data_layer.trajectory_gates import audit_generated_answer, audit_trace_semantics, audit_trajectory_evidence  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows=[]
    with path.open(encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            if not line.strip(): continue
            try: rows.append(json.loads(line))
            except Exception as exc: rows.append({'_bad_json_line':i,'_error':str(exc)})
    return rows


def coordinate_mutated(a: str,b: str)->bool:
    a=re.sub(r'\s+','',a or '').lower(); b=re.sub(r'\s+','',b or '').lower()
    return ('[0,1000]' in a and '[0,1000]' not in b) or ('[0,100]' in a and '[0,100]' not in b)


def newest(paths):
    x=[p for p in paths if p.is_file()]
    return max(x,key=lambda p:p.stat().st_mtime) if x else None


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--settings'); ap.add_argument('--raw',type=Path); ap.add_argument('--raw-dir',type=Path,default=Path('/home/nhl/fxy/results/dota128_pipeline/official_socratic/raw')); ap.add_argument('--api-log',type=Path); ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    raw_path=args.raw or newest(p for p in args.raw_dir.glob('*debug*.jsonl') if not p.name.startswith('local_api_calls_'))
    if not raw_path: raise SystemExit('[FAIL] no debug raw')
    api_path=args.api_log or newest(args.raw_dir.glob('local_api_calls_*_debug_*.jsonl'))
    rows=read_jsonl(raw_path)
    settings=load_settings(settings_from_cli(__file__,args.settings)) if args.settings else None
    index={}
    if settings:
        agent=Path(settings['paths']['pipeline_work_root'])/'agent_inputs'/'train_agent_inputs.jsonl'
        if agent.exists(): index={r['id']:r for r in common_read_jsonl(agent)}

    errors=Counter(); tasks=Counter(); strict=Counter(); success=0; finals=0; mutations=0; bad=0
    duplicate_question_trajectories=0
    classification_contradiction_trajectories=0
    classification_leading_question_trajectories=0
    grounding_gates=Counter()
    strict_rejection_reasons=Counter()
    item_audits=[]
    for row in rows:
        if '_bad_json_line' in row: bad+=1; continue
        task=str(row.get('task','unknown')); tasks[task]+=1
        loop=row.get('loop_result') if isinstance(row.get('loop_result'),dict) else {}
        if loop.get('success'): success+=1
        if loop.get('final_answer'): finals+=1
        err=loop.get('error') or ('verifier_reject' if loop.get('final_answer') and not loop.get('success') else 'none')
        errors[str(err)]+=1
        if coordinate_mutated(str(row.get('query','')),str(row.get('rewritten_query',''))): mutations+=1
        item=index.get(str(row.get('id','')))
        if item:
            geom=audit_trajectory_evidence(item,row,settings) if task=='ref_grounding_obb' else audit_generated_answer(item,str(loop.get('final_answer') or ''),settings)
            sem=audit_trace_semantics(item,row,settings)
            if sem.get('duplicate_question_trajectory'):
                duplicate_question_trajectories+=1
            if task=='ref_classification' and sem.get('classification_contradiction'):
                classification_contradiction_trajectories+=1
            if task=='ref_classification' and sem.get('classification_leading_questions'):
                classification_leading_question_trajectories+=1
            if task=='ref_grounding_obb':
                grounding_gates['iou_pass']+=int(bool(geom.get('iou_gate')))
                grounding_gates['center_pass']+=int(bool(geom.get('center_gate')))
                grounding_gates['both_pass']+=int(bool(geom.get('iou_gate')) and bool(geom.get('center_gate')))
            rounds=sum(1 for turn in (loop.get('chat_history') or []) if turn.get('P_response'))
            min_by_task=(settings or {}).get('trajectory',{}).get(
                'min_perception_rounds_by_task',
                {'ref_grounding_obb':2,'ref_classification':1},
            )
            required_rounds=int(min_by_task.get(task,(settings or {}).get('trajectory',{}).get('min_perception_rounds_strict',2)))
            strict_ok=bool(geom.get('pass')) and bool(sem.get('pass')) and not str(loop.get('error') or '').strip() and rounds>=required_rounds
            rejection=[]
            if not geom.get('pass'): rejection.append('geometry:'+str(geom.get('reason') or 'failed'))
            if not sem.get('pass'): rejection.append('semantics:'+str(sem.get('reason') or 'failed'))
            if str(loop.get('error') or '').strip(): rejection.append('loop_error:'+str(loop.get('error')))
            if rounds<required_rounds: rejection.append(f'perception_rounds:{rounds}<{required_rounds}')
            if strict_ok:
                strict[task]+=1
            else:
                for reason in rejection: strict_rejection_reasons[reason]+=1
            item_audits.append({
                'id':row.get('id'),'task':task,'official_success':bool(loop.get('success')),
                'final_answer':loop.get('final_answer'),'perception_rounds':rounds,
                'required_perception_rounds':required_rounds,'strict_candidate':strict_ok,
                'rejection_reasons':rejection,'geometry':geom,'semantics':sem,
            })

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
        'max_prompt_tokens':None,
        'max_completion_tokens':None,
        'perceiver_calls':0,
        'perceiver_context_missing':0,
        'perceiver_focus_roi_missing':0,
        'classification_perceiver_calls':0,
        'classification_focus_roi_missing':0,
        'focus_crop_missing':0,
        'focus_crop_calls':0,
        'deterministic_verifier_calls':0,
    }
    prompts=[]; completions=[]
    if api_path and api_path.exists():
        for row in read_jsonl(api_path):
            ev=row.get('event')
            if ev=='api_call':
                api['calls']+=1
                if str(row.get('finish_reason','')).lower()=='length': api['length_truncation']+=1
                if isinstance(row.get('prompt_tokens'),int): prompts.append(row['prompt_tokens'])
                if isinstance(row.get('completion_tokens'),int): completions.append(row['completion_tokens'])
                if row.get('role')=='perceiver':
                    api['perceiver_calls']+=1
                    if not row.get('original_context_attached'):
                        api['perceiver_context_missing']+=1
                    if not row.get('focus_roi_attached'):
                        api['perceiver_focus_roi_missing']+=1
                    if 'focus_crop_attached' in row:
                        if row.get('focus_crop_attached'):
                            api['focus_crop_calls']+=1
                        else:
                            api['focus_crop_missing']+=1
                    if row.get('original_context_task')=='ref_classification':
                        api['classification_perceiver_calls']+=1
                        if not row.get('focus_roi_attached'):
                            api['classification_focus_roi_missing']+=1
            elif ev=='deterministic_verifier': api['deterministic_verifier_calls']+=1
            elif ev=='api_error': api['api_errors']+=1
            elif ev=='perceiver_context_missing': api['perceiver_context_missing']+=1
            elif ev=='reasoner_format_repaired':
                api['format_repairs']+=1
                api['reasoner_format_repairs']+=1
            elif ev=='reasoner_format_unrepaired':
                api['unrepaired']+=1
                api['reasoner_format_unrepaired']+=1
                if bool(row.get('fail_closed')):
                    api['unrepaired_fail_closed']+=1
                    api['reasoner_format_unrepaired_fail_closed']+=1
                else:
                    api['unrepaired_unsafe']+=1
            elif ev=='perceiver_response_repaired':
                api['format_repairs']+=1
                api['perceiver_response_repairs']+=1
            elif ev=='perceiver_response_unrepaired':
                api['unrepaired']+=1
                api['perceiver_response_unrepaired']+=1
                if bool(row.get('fail_closed')):
                    api['unrepaired_fail_closed']+=1
                    api['perceiver_response_unrepaired_fail_closed']+=1
                else:
                    api['unrepaired_unsafe']+=1
    api['max_prompt_tokens']=max(prompts) if prompts else None; api['max_completion_tokens']=max(completions) if completions else None
    attempted=sum(tasks.values())
    strict_g=strict.get('ref_grounding_obb',0); total_g=tasks.get('ref_grounding_obb',0)
    strict_c=strict.get('ref_classification',0); total_c=tasks.get('ref_classification',0)
    strict_rate={
        'ref_grounding_obb': round(strict_g/max(1,total_g),6),
        'ref_classification': round(strict_c/max(1,total_c),6),
    }
    trajectory_cfg=(settings or {}).get('trajectory',{})
    min_g=float(trajectory_cfg.get('debug_min_grounding_strict_rate',0.30))
    min_c=float(trajectory_cfg.get('debug_min_classification_strict_rate',0.70))
    geometry_cfg=trajectory_cfg.get('geometry_gate',{})
    config_failures=[]
    if bool(geometry_cfg.get('pass_if_iou_or_center',False)):
        config_failures.append('geometry_gate.pass_if_iou_or_center_must_be_false')
    hard_failures=[]
    if not api_path or not api_path.exists():
        hard_failures.append('api_log_missing')
    elif attempted and not api['perceiver_calls']:
        hard_failures.append('perceiver_calls=0')
    for name,value in (
        ('bad_json_lines',bad),
        ('coordinate_rewrite_mutations',mutations),
        ('length_truncation',api['length_truncation']),
        ('api_errors',api['api_errors']),
        # Exhausted repairs are diagnostic when the adapter replaces the bad
        # output with a deterministic safe fallback. Only an unrepaired response
        # that leaks into the trace remains a hard failure.
        ('unrepaired_unsafe',api['unrepaired_unsafe']),
        ('perceiver_context_missing',api['perceiver_context_missing']),
        ('perceiver_focus_roi_missing',api['perceiver_focus_roi_missing']),
        ('classification_focus_roi_missing',api['classification_focus_roi_missing']),
    ):
        if value:
            hard_failures.append(f'{name}={value}')
    if bool(trajectory_cfg.get('enable_perceiver_focus_crop',False)) and api['perceiver_calls'] and api['focus_crop_missing']:
        hard_failures.append(f"focus_crop_missing={api['focus_crop_missing']}")
    hard_failures.extend(config_failures)
    quality_failures=[]
    if not total_g:
        quality_failures.append('no_grounding_samples')
    elif strict_rate['ref_grounding_obb']<min_g:
        quality_failures.append(f"grounding_strict_rate={strict_rate['ref_grounding_obb']:.3f}<{min_g:.3f}")
    if not total_c:
        quality_failures.append('no_classification_samples')
    elif strict_rate['ref_classification']<min_c:
        quality_failures.append(f"classification_strict_rate={strict_rate['ref_classification']:.3f}<{min_c:.3f}")
    if duplicate_question_trajectories:
        quality_failures.append(f'duplicate_question_trajectories={duplicate_question_trajectories}')
    if classification_contradiction_trajectories:
        quality_failures.append(f'classification_contradiction_trajectories={classification_contradiction_trajectories}')
    if classification_leading_question_trajectories:
        quality_failures.append(f'classification_leading_question_trajectories={classification_leading_question_trajectories}')
    if strict_g>grounding_gates.get('both_pass',0):
        hard_failures.append('strict_grounding_contains_iou_or_center_only_pass')
    status='FAIL' if hard_failures else ('WARN' if quality_failures else 'PASS')
    report={
        'schema_version':'official_debug_quality_gate_v4_3_3',
        'status':status,
        'raw':str(raw_path),
        'attempted':attempted,
        'official_success':success,
        'final_present':finals,
        'tasks':dict(tasks),
        'errors':dict(errors),
        'strict_candidates_by_task':dict(strict),
        'strict_rate_by_task':strict_rate,
        'thresholds':{
            'grounding_strict_rate_min':min_g,
            'classification_strict_rate_min':min_c,
            'duplicate_question_trajectories_max':0,
            'classification_contradiction_trajectories_max':0,
            'classification_leading_question_trajectories_max':0,
        },
        'duplicate_question_trajectories':duplicate_question_trajectories,
        'classification_contradiction_trajectories':classification_contradiction_trajectories,
        'classification_leading_question_trajectories':classification_leading_question_trajectories,
        'grounding_gate_counts':dict(grounding_gates),
        'strict_rejection_reasons':dict(strict_rejection_reasons),
        'item_audit_count':len(item_audits),
        'coordinate_rewrite_mutations':mutations,
        'bad_json_lines':bad,
        'api':api,
        'hard_failures':hard_failures,
        'quality_failures':quality_failures,
    }
    out=args.output or raw_path.with_suffix('.audit.json'); out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    details_path=out.with_name(out.stem+'.items.jsonl')
    details_path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in item_audits),encoding='utf-8')
    csv_path=out.with_name(out.stem+'.items.csv')
    csv_fields=[
        'id','task','official_success','strict_candidate','perception_rounds',
        'required_perception_rounds','final_answer','rejection_reasons',
        'geometry_reason','hbb_iou','iou_gate','center_gate','quadrilateral_valid',
        'focus_copy','final_obb_valid','final_obb_reason','semantic_reasons',
    ]
    with csv_path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=csv_fields)
        writer.writeheader()
        for audit in item_audits:
            geom=audit.get('geometry') if isinstance(audit.get('geometry'),dict) else {}
            sem=audit.get('semantics') if isinstance(audit.get('semantics'),dict) else {}
            writer.writerow({
                'id':audit.get('id'),'task':audit.get('task'),
                'official_success':audit.get('official_success'),
                'strict_candidate':audit.get('strict_candidate'),
                'perception_rounds':audit.get('perception_rounds'),
                'required_perception_rounds':audit.get('required_perception_rounds'),
                'final_answer':audit.get('final_answer'),
                'rejection_reasons':' ; '.join(audit.get('rejection_reasons') or []),
                'geometry_reason':geom.get('reason'),'hbb_iou':geom.get('hbb_iou'),
                'iou_gate':geom.get('iou_gate'),'center_gate':geom.get('center_gate'),
                'quadrilateral_valid':geom.get('quadrilateral_valid'),
                'focus_copy':geom.get('focus_copy'),
                'final_obb_valid':geom.get('final_obb_valid'),
                'final_obb_reason':geom.get('final_obb_reason'),
                'semantic_reasons':' ; '.join(sem.get('reasons') or []),
            })
    print(f'[OFFICIAL DEBUG AUDIT] {status}')
    print(f'  attempted={attempted} official_success={success} final={finals}')
    print(f'  tasks={dict(tasks)} strict_candidates={dict(strict)}')
    print(f'  strict_rates={strict_rate} required=grounding>={min_g:.0%}, classification>={min_c:.0%}')
    print(f'  duplicate_questions={duplicate_question_trajectories} classification_contradictions={classification_contradiction_trajectories} class_leading={classification_leading_question_trajectories}')
    print(f'  grounding_gates={dict(grounding_gates)}')
    print(f'  strict_rejection_reasons={dict(strict_rejection_reasons)}')
    print(f'  errors={dict(errors)}')
    print(f'  coordinate_mutations={mutations} api={api}')
    if hard_failures: print(f'  hard_failures={hard_failures}')
    if quality_failures: print(f'  quality_failures={quality_failures}')
    print(f'  report: {out}')
    print(f'  item details: {details_path}')
    print(f'  item table: {csv_path}')
    return 0 if status=='PASS' else (2 if status=='FAIL' else 3)

if __name__=='__main__': raise SystemExit(main())
