#!/usr/bin/env python3
"""Data-driven sentence composer: no intent rules, no templates.

It retrieves source->target examples, then recombines target sentences by learned
similarity/centrality. This is less fluent than a neural LM, but avoids copying a
whole single target and uses no task-specific branch.
"""
from __future__ import annotations

import argparse, json, math, re, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence, List
from collections import Counter

from emergent_rsg_lm import QwenTokenizer, Featureizer, SparseRetriever, Pair, load_pairs, longest_common_substring_len

SENT_RE = re.compile(r"(?<=[.!?。！？])\s+")

@dataclass
class ComposedResult:
    prompt: str
    output: str
    selected_sentences: List[str]
    neighbors: list
    exact_training_target_match: bool
    longest_training_target_substring_chars: int
    timings: dict

class DataDrivenComposer:
    def __init__(self, tok: QwenTokenizer):
        self.tok = tok
        self.feat = Featureizer(tok)
        self.pairs: List[Pair] = []
        self.retriever: SparseRetriever | None = None

    def fit(self, raw_pairs):
        self.pairs = []
        for s,t in raw_pairs:
            self.pairs.append(Pair(s, t, self.tok.encode(s).ids, self.tok.encode(t).ids, self.feat.features(s)))
        self.retriever = SparseRetriever(self.pairs)
        return self

    def _sim(self, a: Counter[str], b: Counter[str]) -> float:
        assert self.retriever is not None
        return self.retriever.score(a,b)

    def compose(self, prompt: str, k: int = 8) -> ComposedResult:
        assert self.retriever is not None
        t0=time.perf_counter()
        qf=self.feat.features(prompt)
        neigh=self.retriever.retrieve(qf,k=k)
        t1=time.perf_counter()
        if not neigh:
            return ComposedResult(prompt,"",[],[],False,0,{"retrieve_seconds":round(t1-t0,6),"compose_seconds":0.0})
        weights={n.index:n.score for n in neigh}
        # Candidate sentence pool from neighbor targets only.
        cands=[]
        for nrank, n in enumerate(neigh):
            for sent_idx, sent in enumerate(SENT_RE.split(n.target.strip())):
                sent=sent.strip()
                if not sent: continue
                sf=self.feat.features(sent)
                rel=(1.0+self._sim(qf,sf))*(1.0+weights[n.index])
                cands.append({"sentence":sent,"features":sf,"rel":rel,"source_index":n.index,"neighbor_rank":nrank,"sent_idx":sent_idx})
        # Consensus centrality over candidates: sentence preferred when it is close to multiple retrieved targets.
        for c in cands:
            sims=[self._sim(c["features"], d["features"]) for d in cands if d is not c]
            c["centrality"]=(sum(sims)/len(sims)) if sims else 0.0
            c["base"] = c["rel"] * (1.0 + c["centrality"])
        # Learned target length: median target length of retrieved neighbors.
        lengths=sorted(len(self.pairs[n.index].target) for n in neigh)
        target_len=lengths[len(lengths)//2]
        selected=[]; selected_meta=[]; selected_features=[]; used=set(); source_counts={}; total_len=0
        while cands and total_len < target_len and len(selected) < 6:
            best=None; best_score=-1.0
            for c in cands:
                s=c["sentence"]
                if s in used: continue
                redundancy=max((self._sim(c["features"], f) for f in selected_features), default=0.0)
                source_reuse=source_counts.get(c["source_index"],0)
                score=c["base"]/((1.0+redundancy)*(1.0+source_reuse))
                if score>best_score:
                    best_score=score; best=c
            if best is None: break
            selected.append(best["sentence"]); selected_meta.append(best); selected_features.append(best["features"]); used.add(best["sentence"]); source_counts[best["source_index"]]=source_counts.get(best["source_index"],0)+1; total_len += len(best["sentence"])
            cands.remove(best)
        ordered=[m["sentence"] for m in sorted(selected_meta, key=lambda m: (m["sent_idx"], m["neighbor_rank"]))]
        output=" ".join(ordered).strip()
        t2=time.perf_counter()
        neighbor_targets = [self.pairs[n.index].target for n in neigh]
        target_set = {t for t in neighbor_targets}
        return ComposedResult(
            prompt=prompt,
            output=output,
            selected_sentences=ordered,
            neighbors=[{"index":n.index,"score":n.score,"source":n.source,"target":n.target} for n in neigh],
            exact_training_target_match=output in target_set,
            longest_training_target_substring_chars=max((longest_common_substring_len(output, n["target"]) for n in neigh), default=0),
            timings={"retrieve_seconds":round(t1-t0,6),"compose_seconds":round(t2-t1,6)}
        )

def run(args):
    tok=QwenTokenizer(args.tokenizer)
    pairs=load_pairs(args.pairs)
    comp=DataDrivenComposer(tok).fit(pairs)
    res=comp.compose(args.prompt,k=args.k)
    return {"engine":"No-intent data-driven sentence composer over Qwen-BPE features", "tokenizer":tok.analyze(), "training_pairs":len(pairs), "result":asdict(res)}

def main(argv: Sequence[str]|None=None)->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--tokenizer',required=True)
    ap.add_argument('--pairs',nargs='+',required=True)
    ap.add_argument('--prompt',required=True)
    ap.add_argument('--k',type=int,default=8)
    ap.add_argument('--out',default='')
    args=ap.parse_args(argv)
    payload=run(args); text=json.dumps(payload,ensure_ascii=False,indent=2); print(text)
    if args.out: Path(args.out).write_text(text,encoding='utf-8')
    return 0
if __name__=='__main__':
    raise SystemExit(main())
