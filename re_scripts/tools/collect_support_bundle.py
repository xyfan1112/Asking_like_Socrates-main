#!/usr/bin/env python3
"""Create a bounded, redacted support bundle without model weights or images."""
from __future__ import annotations
import argparse,json,os,re,shutil,subprocess,tarfile,tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

TEXT_EXT={'.json','.jsonl','.txt','.log','.md','.yaml','.yml','.sh','.py','.csv'}
KEY_SECRET_RE=re.compile(
    r'''(?ix)(["']?(?:api[_-]?key|authorization|password|secret|access[_-]?token|refresh[_-]?token)["']?\s*[:=]\s*)(["']?)([^"',}\s]+)(["']?)'''
)
BEARER_RE=re.compile(r'(?i)\bBearer\s+[A-Za-z0-9._~+\-/=]+')
EXCLUDE_NAMES={'adapter_model.safetensors','adapter_model.bin','model.safetensors','pytorch_model.bin'}


def safe_rel(path:Path,roots:list[Path])->Path:
    for root in roots:
        try: return Path(root.name)/path.resolve().relative_to(root.resolve())
        except Exception: pass
    return Path('misc')/path.name


def redact(text:str)->str:
    text=BEARER_RE.sub('Bearer <REDACTED>',text)
    return KEY_SECRET_RE.sub(lambda m:m.group(1)+(m.group(2) or '')+'<REDACTED>'+(m.group(4) if m.group(4)==m.group(2) else ''),text)


def copy_text(src:Path,dst:Path,max_bytes:int,sample_rows:int)->None:
    dst.parent.mkdir(parents=True,exist_ok=True)
    size=src.stat().st_size
    if src.suffix=='.jsonl' and size>max_bytes:
        lines=src.read_text(encoding='utf-8',errors='replace').splitlines()
        picked=lines[:sample_rows]+(['{"_support_bundle_note":"middle rows omitted"}'] if len(lines)>sample_rows*2 else [])+lines[-sample_rows:]
        dst.write_text(redact('\n'.join(picked)+'\n'),encoding='utf-8')
    elif size<=max_bytes:
        dst.write_text(redact(src.read_text(encoding='utf-8',errors='replace')),encoding='utf-8')
    else:
        with src.open('rb') as f:
            head=f.read(max_bytes//2); f.seek(max(0,size-max_bytes//2)); tail=f.read(max_bytes//2)
        dst.write_text(redact(head.decode('utf-8','replace')+'\n\n[...TRUNCATED...]\n\n'+tail.decode('utf-8','replace')),encoding='utf-8')


def command(out:Path,name:str,cmd:list[str])->None:
    try: text=subprocess.check_output(cmd,stderr=subprocess.STDOUT,text=True,timeout=30)
    except Exception as e: text=f'{type(e).__name__}: {e}\n'
    (out/name).write_text(redact(text),encoding='utf-8')


def iter_files(roots:Iterable[Path],mode:str)->Iterable[Path]:
    keywords={
      'data':['report','audit','build','lineage','classes','config','runtime_settings'],
      'socratic':['socratic','trajectory','agent','truncation','debug','raw','postproc'],
      'train-b1':['b1','direct','train','config','contract','trainer','loss'],
      'train-b2':['b2','socratic','train','config','contract','trainer','loss'],
      'eval-b0':['rs_eot','matrix','metric','pred','server'],
      'eval-b1':['b1','matrix','metric','pred','server'],
      'eval-b2':['b2','matrix','metric','pred','server'],
      'error':['log','report','status','config','runtime','error'],
      'all':[],
    }[mode]
    for root in roots:
        if not root.exists(): continue
        for path in root.rglob('*'):
            if not path.is_file() or path.name in EXCLUDE_NAMES or path.suffix.lower() not in TEXT_EXT: continue
            lower=str(path).lower()
            if mode!='all' and not any(k in lower for k in keywords): continue
            yield path


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--settings',required=True,type=Path)
    ap.add_argument('--repo-root',required=True,type=Path)
    ap.add_argument('--mode',choices=['error','data','socratic','train-b1','train-b2','eval-b0','eval-b1','eval-b2','all'],required=True)
    ap.add_argument('--max-log-mb',type=int,default=20)
    ap.add_argument('--sample-rows',type=int,default=20)
    args=ap.parse_args()
    s=json.loads(args.settings.read_text(encoding='utf-8'))
    p=s['paths']; output_root=Path(p['results_root']).parent
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    bundle_dir=output_root/'support_bundles'; bundle_dir.mkdir(parents=True,exist_ok=True)
    tar_path=bundle_dir/f'support_bundle_{s.get("taxonomy",{}).get("qa_language","unknown")}_{args.mode}_{stamp}.tar.gz'
    with tempfile.TemporaryDirectory(prefix='v123_support_') as td:
        stage=Path(td)/tar_path.stem.replace('.tar',''); stage.mkdir()
        meta=stage/'meta'; meta.mkdir()
        shutil.copy2(args.settings,meta/'resolved_settings.json')
        for name in ['PATCH_VERSION.json','SSH_VERSION.json','VERSION.json','SHA256SUMS','PACKAGE_MANIFEST.json']:
            src=args.repo_root/name
            if src.is_file(): shutil.copy2(src,meta/name)
        command(meta,'nvidia_smi.txt',['nvidia-smi'])
        command(meta,'nvidia_smi_list.txt',['nvidia-smi','-L'])
        command(meta,'disk.txt',['df','-h'])
        command(meta,'processes.txt',['bash','-lc','ps -ef | grep -E "vllm|llamafactory|torchrun|generation.py" | grep -v grep || true'])
        roots=[args.repo_root,Path(p['pipeline_work_root']),Path(p['training_run_root']),Path(p['test_run_root']),Path(p['dota128_llamafactory_root']),Path(p['results_root'])]
        seen=set(); copied=[]; max_bytes=args.max_log_mb*1024*1024
        for src in iter_files(roots,args.mode):
            key=str(src.resolve())
            if key in seen: continue
            seen.add(key)
            rel=safe_rel(src,roots)
            try: copy_text(src,stage/'files'/rel,max_bytes,args.sample_rows); copied.append(str(src))
            except Exception as e:
                (stage/'copy_errors.txt').open('a',encoding='utf-8').write(f'{src}: {type(e).__name__}: {e}\n')
        manifest={'schema_version':'support_bundle_v1_2_3','mode':args.mode,'settings':str(args.settings),'copied_files':copied,'excluded':['model weights','adapter weights','images','files with unsupported binary extensions'],'redaction':'common key/token/password patterns'}
        (stage/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        with tarfile.open(tar_path,'w:gz') as tf: tf.add(stage,arcname=stage.name)
    print(f'[PASS] support bundle: {tar_path}')
    return 0
if __name__=='__main__': raise SystemExit(main())
