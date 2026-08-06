#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'main_layer'))
from common import load_settings,settings_from_cli
from config import resolve_python


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--settings');args=ap.parse_args()
 s=load_settings(settings_from_cli(__file__,args.settings)); rl=s['reinforcement_learning']; easy=Path(s['project']['easy_r1_dir']); py=resolve_python(s,'rl')
 stage=rl['stage1']; model=Path(s['models'][rl['base_model_key']]['path']); data=Path(s['paths']['dota128_rl_root'])
 report={'python':py,'easy_r1_dir':str(easy),'model':str(model),'checks':{},'warnings':[],'critical':[]}
 checks=report['checks']; checks['easy_r1_exists']=(easy/'verl').is_dir(); checks['config_exists']=(easy/'examples/config.yaml').is_file(); checks['model_config']=(model/'config.json').is_file(); checks['train_data']=(data/'train.jsonl').is_file(); checks['val_data']=(data/'val.jsonl').is_file()
 for k,v in checks.items():
  if not v: report['critical'].append(k)
 try:
  out=subprocess.check_output([py,'-c','import torch,verl; print(torch.cuda.device_count())'],text=True,cwd=easy).strip(); checks['gpu_count']=int(out.splitlines()[-1])
  if checks['gpu_count']<int(stage.get('n_gpus_per_node',2)): report['critical'].append('insufficient_gpu_count')
 except Exception as e: report['critical'].append(f'rl_python_import:{type(e).__name__}:{e}')
 has_lora=any('lora' in p.read_text(errors='ignore').lower() for p in (easy/'verl').rglob('*.py')) if (easy/'verl').is_dir() else False
 checks['bundled_easy_r1_lora_support_detected']=has_lora
 if stage.get('training_strategy','').startswith('lora') and not has_lora: report['critical'].append('bundled_EasyR1_has_no_detected_LoRA_support')
 if stage.get('training_strategy')=='fsdp_full_offload': report['warnings'].append('2xA6000 运行7B GRPO仍需先做小batch smoke；显存可行性不能仅靠静态配置保证。')
 report['passed']=not report['critical']; print(json.dumps(report,ensure_ascii=False,indent=2)); raise SystemExit(0 if report['passed'] else 2)
if __name__=='__main__':main()
