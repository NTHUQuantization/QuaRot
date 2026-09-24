"""Windowed R9700 runner for exact projection-only PARD-2 training."""
from __future__ import annotations
import argparse, gc, os, json, math, random, time
from pathlib import Path

import torch
import torch.nn.functional as F

from e2e.qwen3_32b_td_features import atomic_json, gpu_snapshot, preflight_gpu, sha256_file
from e2e.qwen3_32b_td_projection_train import (
    SCALE, build_expanded, cosine_lr, exact_chunked_loss_and_hidden_grad,
    load_cache, load_draft, load_initial_projection, load_target_head,
    save_checkpoint,
)


def windows(ids, feature, prob, loss_start, size):
    for start in range(0, len(ids), size):
        end=min(start+size,len(ids)); local=max(0,min(end-start,loss_start-start))
        if end-start < 8 or local >= end-start-1: continue
        yield ids[start:end],feature[start:end],prob[start:end],local,start


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",required=True);p.add_argument("--draft",required=True);p.add_argument("--target-source",required=True)
    p.add_argument("--calibration",type=Path,required=True);p.add_argument("--train-manifest",type=Path,required=True)
    p.add_argument("--validation-manifest",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--resume",type=Path);p.add_argument("--total-units",type=int,required=True)
    p.add_argument("--gradient-accumulation",type=int,default=8);p.add_argument("--learning-rate",type=float,default=3e-5);p.add_argument("--vocab-chunk",type=int,default=512)
    p.add_argument("--max-context",type=int,default=128);p.add_argument("--seed",type=int,default=20260830)
    p.add_argument("--ce-alpha",type=float,default=.1);p.add_argument("--kd-alpha",type=float,default=1.)
    a=p.parse_args(argv);random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    preflight=preflight_gpu();draft=load_draft(a.draft);target_head=load_target_head(a.target_source)
    initial,raw_scale,raw_bias=load_initial_projection(a.draft,a.calibration)
    projection=torch.nn.Parameter(initial.to("cuda",dtype=torch.float32))
    optimizer=torch.optim.AdamW([projection],lr=3e-5,betas=(.9,.95),weight_decay=0.)
    state={"global_unit":0,"optimizer_step":0,"source_tokens_seen":0,"trained_tokens_seen":0}
    if a.resume:
        r=torch.load(a.resume,map_location="cpu",weights_only=False);projection.data.copy_(r["projection"].to("cuda",dtype=projection.dtype))
        optimizer.load_state_dict(r["optimizer"]);state.update(r["state"])
    raw_scale=raw_scale.to("cuda",dtype=torch.float32);raw_bias=raw_bias.to("cuda",dtype=torch.float32)
    snapshots=[gpu_snapshot("runtime_loaded")]

    def one(ids,feature,prob,loss_start,seed,training):
        calibrated=feature.to("cuda",dtype=torch.float32)*raw_scale+raw_bias;gold=prob.to("cuda",dtype=torch.float32)
        ex=build_expanded(ids,calibrated,gold,loss_start,seed=seed,cod=True)
        keep=(random.Random(seed+991).random()>.1) if training else True
        word=draft.get_input_embeddings()(ex["input_ids"]).detach()
        projected=F.linear(ex["target_feat"].to(torch.bfloat16),projection.to(torch.bfloat16))*SCALE
        if not training: projected=projected.detach()
        if not keep: projected=projected*0
        with torch.set_grad_enabled(training):
            hidden=draft.model(input_ids=None,inputs_embeds=(word+projected)[None],attention_mask=ex["attention_mask"],
                position_ids=ex["position_ids"][None],use_cache=False,return_dict=True).last_hidden_state
        shifted=hidden[0,:-1];labels=ex["labels"][1:];weight=ex["prev_prob"][1:]
        teacher=feature.to("cuda",dtype=torch.bfloat16)[ex["teacher_index"][:-1],:5120]
        loss,ce,kd,grad=exact_chunked_loss_and_hidden_grad(shifted.detach(),teacher,labels,weight,
            draft.lm_head.weight.detach(),target_head,ce_alpha=a.ce_alpha,kd_alpha=a.kd_alpha,chunk_size=a.vocab_chunk)
        if not torch.isfinite(torch.stack((loss,ce,kd))).all() or not torch.isfinite(grad).all(): raise FloatingPointError(f"non-finite loss/gradient seed={seed}")
        if training and keep: shifted.backward(grad.to(shifted.dtype))
        return float(loss),float(ce),float(kd),int(keep),len(ids),int(shifted.shape[0])

    train_meta=json.loads(a.train_manifest.read_text());data_path=Path(train_meta["data"])
    wanted={r["ordinal"] for payload in load_cache(a.train_manifest) for r in payload["rows"]};ids_map={}
    with data_path.open(encoding="utf-8") as h:
        for ordinal,line in enumerate(h):
            if ordinal in wanted: ids_map[ordinal]=json.loads(line)["input_ids"]

    def evaluate():
        draft.eval();values=[]
        for payload in load_cache(a.validation_manifest):
            off=0
            for row in payload["rows"]:
                n=int(row["tokens"]);ids=row.get("input_ids")
                if ids is None: raise RuntimeError("validation teacher row lacks input_ids")
                feat=payload["features"][0,off:off+n];prob=payload["teacher_gold_prob"][0,off:off+n]
                for wi,(x,f,g,l,s) in enumerate(windows(ids,feat,prob,int(row["loss_start"]),a.max_context)):
                    values.append(one(x,f,g,l,a.seed+row["ordinal"]*100+wi,False)[:3])
                off+=n
        draft.train();return {"loss":sum(x[0] for x in values)/len(values),"ce":sum(x[1] for x in values)/len(values),
            "kd":sum(x[2] for x in values)/len(values),"units":len(values)}

    before=evaluate();started=time.perf_counter();history=[];pending=0;stage_source=0;stage_trained=0
    for payload in load_cache(a.train_manifest):
        off=0
        for row in payload["rows"]:
            n=int(row["tokens"]);ids=ids_map[row["ordinal"]];feat=payload["features"][0,off:off+n];prob=payload["teacher_gold_prob"][0,off:off+n]
            stage_source+=n
            for wi,(x,f,g,l,s) in enumerate(windows(ids,feat,prob,int(row["loss_start"]),a.max_context)):
                z=one(x,f,g,l,a.seed+row["ordinal"]*100+wi,True);pending+=1;state["global_unit"]+=1
                state["trained_tokens_seen"]+=z[4];stage_trained+=z[4]
                history.append({"unit":state["global_unit"],"loss":z[0],"ce":z[1],"kd":z[2],"keep":z[3],"tokens":z[4],"expanded":z[5]})
                if pending>=a.gradient_accumulation:
                    projection.grad.div_(pending);assert torch.isfinite(projection.grad).all(), "non-finite projection gradient";lr=cosine_lr(state["global_unit"],a.total_units,base_lr=a.learning_rate)
                    for group in optimizer.param_groups:group["lr"]=lr
                    torch.nn.utils.clip_grad_norm_([projection],1.);optimizer.step();optimizer.zero_grad(set_to_none=True)
                    state["optimizer_step"]+=1;pending=0
                if state["global_unit"]%32==0:snapshots.append(gpu_snapshot(f"unit_{state['global_unit']}"))
            off+=n
        gc.collect();torch.cuda.empty_cache()
    state["source_tokens_seen"]+=stage_source
    if pending:
        projection.grad.div_(pending);assert torch.isfinite(projection.grad).all(), "non-finite projection gradient";lr=cosine_lr(state["global_unit"],a.total_units,base_lr=a.learning_rate);[group.update(lr=lr) for group in optimizer.param_groups];torch.nn.utils.clip_grad_norm_([projection],1.);optimizer.step();optimizer.zero_grad(set_to_none=True);state["optimizer_step"]+=1
    elapsed=time.perf_counter()-started;after=evaluate();snapshots.append(gpu_snapshot("complete"))
    artifacts=save_checkpoint(a.output_dir,projection,optimizer,state,a.draft)
    result={"schema_version":1,"stage":a.stage,"method":"windowed-128-exact","trainable_parameters":projection.numel(),
        "fixed":{"max_context":a.max_context,"para_num":16,"scale":.02,"ce_alpha":a.ce_alpha,"kd_alpha":a.kd_alpha,"temperature":1.,
            "target_feat_mask":.1,"cod":[.7,.1],"vocab_chunk":a.vocab_chunk,"gradient_checkpointing":True,"optimizer_precision":"fp32_master","learning_rate":a.learning_rate},
        "train_manifest":str(a.train_manifest.resolve()),"train_manifest_sha256":sha256_file(a.train_manifest),
        "validation_manifest":str(a.validation_manifest.resolve()),"validation_before":before,"validation_after":after,
        "validation_improved":after["loss"]<before["loss"],"state":state,"stage_source_tokens":stage_source,
        "stage_trained_tokens":stage_trained,"elapsed_seconds":elapsed,"trained_tokens_per_second":stage_trained/elapsed,
        "history_tail":history[-32:],"artifacts":artifacts,"gpu_preflight":preflight,"memory_snapshots":snapshots}
    atomic_json(a.output_dir/"train_result.json",result);print(json.dumps({k:v for k,v in result.items() if k not in {"history_tail","memory_snapshots"}},indent=2))


if __name__=="__main__":main()
