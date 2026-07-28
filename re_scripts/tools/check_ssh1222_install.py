#!/usr/bin/env python3
from pathlib import Path
import json,sys
root=Path(__file__).resolve().parents[1]
settings=Path(sys.argv[1]) if len(sys.argv)>1 else root/'settings.json'
s=json.load(open(settings,encoding='utf-8'))
checks={
 'ssh_version':json.load(open(root/'SSH_VERSION.json',encoding='utf-8')).get('ssh_version')=='1.2.2.2',
 'perceiver_512':int(s['agents']['perceiver']['max_tokens'])==512,
 'similarity_095':float(s['trajectory']['question_similarity_threshold'])==0.95,
 'classification_one_round':int(s['trajectory']['min_perception_rounds_by_task']['ref_classification'])==1,
 'resume_script':(root/'main_layer/resume_official_artifacts.sh').is_file(),
 'command_registered':'resume-official-artifacts' in (root/'main_layer/run.py').read_text(encoding='utf-8'),
 'safe_truncation_audit':'recovered_length_truncation' in (root/'data_layer/official_socratic/02c_audit_full_generation.py').read_text(encoding='utf-8'),
}
for k,v in checks.items(): print(f"[{'OK' if v else 'FAIL'}] {k}")
raise SystemExit(0 if all(checks.values()) else 2)
