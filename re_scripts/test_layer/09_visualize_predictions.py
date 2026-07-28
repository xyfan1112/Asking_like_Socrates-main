#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,textwrap
from pathlib import Path
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'main_layer'))
from common import read_jsonl,obb_iou

def poly(draw,pts,color,width=3):
 xy=[tuple(map(lambda v:int(round(v)),p)) for p in pts]
 if len(xy)==4:draw.line(xy+[xy[0]],fill=color,width=width)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pred',required=True);ap.add_argument('--task',choices=['raw','ref'],required=True);ap.add_argument('--output-dir',required=True);ap.add_argument('--limit',type=int,default=200);args=ap.parse_args();rows=read_jsonl(args.pred,skip_bad=True);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
 for i,r in enumerate(rows[:args.limit]):
  im=Image.open(r['image_path']).convert('RGB');draw=ImageDraw.Draw(im);w=max(2,min(im.size)//250)
  if args.task=='raw':
   for x in r.get('ground_truth',[]):poly(draw,x['points'],(0,220,0),w)
   for x in r.get('predictions',[]):poly(draw,x['points'],(255,0,0),w)
   title=f"{r.get('model_key')} run={r.get('run_id')} GT={len(r.get('ground_truth',[]))} Pred={len(r.get('predictions',[]))}"
  else:
   gt_points=r.get('gt_obb') or r.get('gt_points') or []
   pred_points=r.get('pred_obb') or r.get('pred_points')
   poly(draw,gt_points,(0,220,0),w)
   if pred_points:poly(draw,pred_points,(255,0,0),w)
   iou=obb_iou(pred_points,gt_points) if pred_points else 0
   title=f"run={r.get('run_id')} IoU={iou:.3f} GT={r.get('gt_class')} Pred={r.get('pred_class')}"
  canvas=Image.new('RGB',(im.width,im.height+45),'white');canvas.paste(im,(0,45));ImageDraw.Draw(canvas).text((5,5),title,fill='black');canvas.save(out/f"{i:05d}_{r.get('run_id',0)}.png")
 print(f'保存 {min(len(rows),args.limit)} 张到 {out}；绿色 GT，红色 Pred')
if __name__=='__main__':main()
