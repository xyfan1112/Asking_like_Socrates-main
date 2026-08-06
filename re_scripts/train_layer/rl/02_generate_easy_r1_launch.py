#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,shlex,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'main_layer'))
from common import load_settings,settings_from_cli
from config import resolve_python


def q(x): return shlex.quote(str(x))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--settings');ap.add_argument('--reward',choices=['hbb','obb'],default='hbb');args=ap.parse_args()
 s=load_settings(settings_from_cli(__file__,args.settings)); rl=s['reinforcement_learning']; st=rl['stage1']; easy=Path(s['project']['easy_r1_dir']); py=resolve_python(s,'rl'); model=s['models'][rl['base_model_key']]['path']; data=Path(s['paths']['dota128_rl_root']); run=Path(s['paths']['rl_run_root']); run.mkdir(parents=True,exist_ok=True)
 if args.reward=='obb' and not rl.get('experimental_obb_reward',{}).get('enabled',False):
  raise SystemExit('Experimental OBB reward is disabled in settings. Enable it explicitly for an ablation.')
 prompt=ROOT/'train_layer/rl/prompts'/('dota_grounding_hbb.jinja' if args.reward=='hbb' else 'dota_grounding_obb_experimental.jinja')
 reward=ROOT/'train_layer/rl/rewards'/('dota_hbb_reward.py' if args.reward=='hbb' else 'dota_obb_reward_experimental.py')
 exp=f"dota128_{args.reward}_grounding_grpo"
 script=run/f'launch_{exp}.sh'
 lines=[
 '#!/usr/bin/env bash','set -euo pipefail','set -x',f'cd {q(easy)}',
 'export PYTHONUNBUFFERED=1','export NCCL_ASYNC_ERROR_HANDLING=1','export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64',
 f'CUDA_VISIBLE_DEVICES={q(s["runtime"]["gpu_profiles"]["rl_dual_a6000"]["visible_devices"])} {q(py)} -m verl.trainer.main \\',
 '  config=examples/config.yaml \\',
 f'  data.train_files={q(data/"train.jsonl")} \\',f'  data.val_files={q(data/"val.jsonl")} \\',
 '  data.image_dir=null \\','  data.prompt_key=problem \\','  data.answer_key=answer \\','  data.image_key=images \\',
 f'  data.format_prompt={q(prompt)} \\',f'  data.max_prompt_length={st["max_prompt_length"]} \\',f'  data.max_response_length={st["max_response_length"]} \\',
 f'  data.rollout_batch_size={st["rollout_batch_size"]} \\',f'  data.val_batch_size={st["rollout_batch_size"]} \\',f'  data.min_pixels={st["min_pixels"]} \\',f'  data.max_pixels={st["max_pixels"]} \\',
 f'  worker.actor.model.model_path={q(model)} \\','  worker.actor.model.enable_gradient_checkpointing=true \\','  worker.actor.model.freeze_vision_tower=false \\',
 f'  worker.actor.optim.lr={st["learning_rate"]} \\',f'  worker.actor.global_batch_size={st["actor_global_batch_size"]} \\',f'  worker.actor.micro_batch_size_per_device_for_update={st["micro_batch_size_per_device_for_update"]} \\',f'  worker.actor.micro_batch_size_per_device_for_experience={st["micro_batch_size_per_device_for_experience"]} \\',
 '  worker.actor.offload.offload_params=true \\','  worker.actor.offload.offload_optimizer=true \\','  worker.ref.offload.offload_params=true \\',
 f'  worker.rollout.n={st["rollout_n"]} \\',f'  worker.rollout.tensor_parallel_size={st["rollout_tensor_parallel_size"]} \\',f'  worker.rollout.gpu_memory_utilization={st["rollout_gpu_memory_utilization"]} \\',
 '  worker.reward.reward_type=batch \\',f'  worker.reward.reward_function={q(str(reward)+":compute_score")} \\',
 f'  trainer.experiment_name={exp} \\',f'  trainer.n_gpus_per_node={st["n_gpus_per_node"]} \\',f'  trainer.total_epochs={st["total_epochs"]} \\',f'  trainer.save_checkpoint_path={q(run/exp)}'
 ]
 script.write_text('\n'.join(lines)+'\n',encoding='utf-8');script.chmod(0o755)
 print(json.dumps({'launch_script':str(script),'reward':args.reward,'base_model':model,'training_strategy':st.get('training_strategy'),'important':'Run 00_preflight_rl.py and a reduced rollout_batch_size smoke before a full job.'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
