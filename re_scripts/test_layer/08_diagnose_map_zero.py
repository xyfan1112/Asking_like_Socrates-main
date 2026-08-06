#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'main_layer'))
from common import *

def load_rows(path):
 p=Path(path)
 if p.is_dir():return [json.loads(x.read_text(encoding='utf-8')) for x in sorted(p.glob('*_result.json'))]
 return read_jsonl(p,skip_bad=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--settings');ap.add_argument('--input',required=True);ap.add_argument('--output');args=ap.parse_args();s=load_settings(settings_from_cli(__file__,args.settings));rows=load_rows(args.input);report={'rows':len(rows),'pred_total':0,'gt_total':0,'rows_no_predictions':0,'rows_with_error':0,'coord_modes':Counter(),'best_iou_any_class':[],'best_iou_same_class':[],'likely_causes':[]}
 for r in rows:
  gt=r.get('ground_truth',[]);pred=r.get('predictions',[]);report['gt_total']+=len(gt);report['pred_total']+=len(pred);report['rows_no_predictions']+=int(not pred);report['rows_with_error']+=int(bool(r.get('error')))
  for p in pred:report['coord_modes'][p.get('coordinate_mode_detected','unknown')]+=1
  for p in pred:
   any_i=max([obb_iou(p['points'],g['points']) for g in gt] or [0]);same_i=max([obb_iou(p['points'],g['points']) for g in gt if p.get('class_id')==g.get('class_id')] or [0]);report['best_iou_any_class'].append(any_i);report['best_iou_same_class'].append(same_i)
 report['coord_modes']=dict(report['coord_modes']);report['mean_best_iou_any_class']=sum(report['best_iou_any_class'])/max(len(report['best_iou_any_class']),1);report['mean_best_iou_same_class']=sum(report['best_iou_same_class'])/max(len(report['best_iou_same_class']),1)
 if report['pred_total']==0:report['likely_causes']+=['模型最终区没有可解析 OBB；检查 </think>、FINAL_DETECTIONS、输出截断和解析器。']
 if report['rows_no_predictions']/max(report['rows'],1)>.5:report['likely_causes']+=['超过一半图像零预测，mAP=0 首先是生成/解析失败，不是 IoU 算法。']
 if report['mean_best_iou_any_class']<.1 and report['pred_total']>0:report['likely_causes']+=['预测框与 GT 几乎不重合；重点检查 0-1/0-100/0-1000/像素坐标制和图像缩放。']
 if report['mean_best_iou_any_class']>.3 and report['mean_best_iou_same_class']<.1:report['likely_causes']+=['框位置可能接近，但类别名或 class_id 映射错误。']
 if report['gt_total'] and report['pred_total'] and not report['likely_causes']:report['likely_causes']+=['检查验证类别是否在训练中出现、训练样本量、优化步数及任务 prompt 是否与训练一致。']
 del report['best_iou_any_class'];del report['best_iou_same_class'];out=Path(args.output) if args.output else Path(s['paths']['test_run_root'])/'map_zero_diagnosis.json';write_json(out,report);print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
