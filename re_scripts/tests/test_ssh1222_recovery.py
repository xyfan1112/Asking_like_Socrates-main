from pathlib import Path
import json, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]

def test_settings_preserved():
    s=json.load(open(ROOT/'settings.json',encoding='utf-8'))
    assert s['agents']['perceiver']['max_tokens']==512
    assert s['trajectory']['question_similarity_threshold']==0.95
    assert s['trajectory']['min_perception_rounds_by_task']['ref_classification']==1

def test_resume_command_registered():
    assert 'resume-official-artifacts' in (ROOT/'main_layer/run.py').read_text(encoding='utf-8')

def _audit(tmp_path, unsafe):
    raw=tmp_path/'raw.jsonl'
    raw.write_text(json.dumps({'task':'ref_classification','loop_result':{'success':True,'final_answer':'plane','chat_history':[]}})+'\n',encoding='utf-8')
    log=tmp_path/'api.jsonl'
    rows=[
      {'event':'api_call','role':'perceiver','finish_reason':'length','original_context_attached':True,'focus_roi_attached':True,'original_context_task':'ref_classification'},
      {'event':'reasoner_format_unrepaired','fail_closed':not unsafe},
    ]
    log.write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf-8')
    return subprocess.run([sys.executable,str(ROOT/'data_layer/official_socratic/02c_audit_full_generation.py'),'--raw',str(raw),'--api-log',str(log)],capture_output=True,text=True)

def test_recovered_truncation_is_warning(tmp_path):
    r=_audit(tmp_path,False)
    assert r.returncode==0, r.stdout+r.stderr
    report=json.load(open(tmp_path/'raw.full_audit.json',encoding='utf-8'))
    assert report['status']=='PASS_WITH_WARNINGS'
    assert report['api']['unrepaired_unsafe']==0

def test_unsafe_unrepaired_still_fails(tmp_path):
    r=_audit(tmp_path,True)
    assert r.returncode==2
    report=json.load(open(tmp_path/'raw.full_audit.json',encoding='utf-8'))
    assert report['status']=='FAIL'
    assert report['api']['unrepaired_unsafe']==1
