#!/usr/bin/env python3
"""Single-image grounding/detection CLI with saved visualization."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'main_layer'));sys.path.insert(0,str(ROOT/'test_layer'))
from common import image_data_url,load_settings,parse_hbb_output,parse_obb_output,settings_from_cli
from eval_common import chat_once,check_server,protocol

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--settings');ap.add_argument('--model-key',default='rs_eot');ap.add_argument('--base-url',default='http://127.0.0.1:8010/v1');ap.add_argument('--image',required=True);ap.add_argument('--question',required=True);ap.add_argument('--target',choices=['obb','hbb','text'],default='obb');ap.add_argument('--output-dir');args=ap.parse_args()
 s=load_settings(settings_from_cli(__file__,args.settings));m=s['models'][args.model_key];cfg=protocol(s,'ref_grounding');check_server(args.base_url,m['served_name']);path=Path(args.image).expanduser().resolve();image=Image.open(path).convert('RGB');w,h=image.size
 r=chat_once(args.base_url,m['served_name'],[{'role':'user','content':[{'type':'image_url','image_url':{'url':image_data_url(path)}},{'type':'text','text':args.question}]}],temperature=float(cfg['temperature']),top_p=float(cfg['top_p']),max_tokens=int(cfg['max_tokens']),retries=1)
 draw=ImageDraw.Draw(image);parsed=None
 if args.target=='obb':
  parsed=parse_obb_output(r['raw_output'],w,h,cfg.get('coordinate_mode','normalized_0_1000'))
  for i,item in enumerate(parsed):
   pts=[tuple(x) for x in item['points']];draw.line(pts+[pts[0]],fill='red',width=4);draw.text(pts[0],f"{i}:{item['class_name']}",fill='red')
 elif args.target=='hbb':
  parsed=parse_hbb_output(r['raw_output'],w,h,cfg.get('coordinate_mode','normalized_0_1000'))
  if parsed:
   draw.rectangle(parsed['bbox'],outline='red',width=4);draw.text((parsed['bbox'][0],parsed['bbox'][1]),str(parsed.get('class_name') or 'target'),fill='red')
 out_dir=Path(args.output_dir) if args.output_dir else Path(s['paths']['test_run_root'])/'interactive';out_dir.mkdir(parents=True,exist_ok=True);stem=f"{path.stem}_{args.model_key}";vis=out_dir/f'{stem}_visualized.png';js=out_dir/f'{stem}_result.json';image.save(vis);js.write_text(json.dumps({'image':str(path),'question':args.question,'target':args.target,'raw_output':r['raw_output'],'parsed':parsed,'error':r['error'],'visualization':str(vis)},ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'result':str(js),'visualization':str(vis),'raw_output':r['raw_output'],'parsed':parsed},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
