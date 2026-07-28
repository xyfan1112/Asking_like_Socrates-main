#!/usr/bin/env python3
"""Optional local Gradio window for visual grounding experiments."""
from __future__ import annotations
import argparse,json,sys,tempfile
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'main_layer'));sys.path.insert(0,str(ROOT/'test_layer'))
from common import image_data_url,load_settings,parse_hbb_output,parse_obb_output,settings_from_cli
from eval_common import chat_once,protocol

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--settings');ap.add_argument('--host',default='127.0.0.1');ap.add_argument('--port',type=int,default=7860);args=ap.parse_args();s=load_settings(settings_from_cli(__file__,args.settings))
 try:import gradio as gr
 except ImportError:raise SystemExit('Install optional dependency in the client environment: python -m pip install gradio')
 keys=list(s['evaluation']['model_keys']);cfg=protocol(s,'ref_grounding')
 def run(image,question,model_key,base_url,target):
  if image is None:return None,'No image',None
  path=Path(tempfile.mkstemp(suffix='.png')[1]);Image.fromarray(image).save(path);im=Image.open(path).convert('RGB');w,h=im.size;m=s['models'][model_key]
  r=chat_once(base_url,m['served_name'],[{'role':'user','content':[{'type':'image_url','image_url':{'url':image_data_url(path)}},{'type':'text','text':question}]}],temperature=float(cfg['temperature']),top_p=float(cfg['top_p']),max_tokens=int(cfg['max_tokens']),retries=1);draw=ImageDraw.Draw(im);parsed=None
  if target=='OBB':
   parsed=parse_obb_output(r['raw_output'],w,h,cfg.get('coordinate_mode','normalized_0_1000'))
   for x in parsed:pts=[tuple(p) for p in x['points']];draw.line(pts+[pts[0]],fill='red',width=4)
  elif target=='HBB':
   parsed=parse_hbb_output(r['raw_output'],w,h,cfg.get('coordinate_mode','normalized_0_1000'))
   if parsed:draw.rectangle(parsed['bbox'],outline='red',width=4)
  return im,r['raw_output'],json.dumps(parsed,ensure_ascii=False,indent=2)
 with gr.Blocks(title='DOTA/VRSBench Grounding Debug') as demo:
  gr.Markdown('# 单张遥感图像定位调试')
  with gr.Row():inp=gr.Image();out=gr.Image()
  q=gr.Textbox(label='Question');model=gr.Dropdown(keys,value=keys[0],label='Model key');url=gr.Textbox(value='http://127.0.0.1:8010/v1',label='OpenAI-compatible URL');target=gr.Radio(['OBB','HBB','Text'],value='OBB');btn=gr.Button('Run');raw=gr.Textbox(label='Raw output');parsed=gr.Code(label='Parsed result',language='json');btn.click(run,[inp,q,model,url,target],[out,raw,parsed])
 demo.launch(server_name=args.host,server_port=args.port)
if __name__=='__main__':main()
