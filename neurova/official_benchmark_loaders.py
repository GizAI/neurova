from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional
import csv, json, re
from .external_benchmarks import SCANInterpreter, BabiMicroWorld, KinshipReasoner, BenchmarkCase, BenchmarkReport

@dataclass
class OfficialLoadReport:
    benchmark: str
    path: str
    loaded: bool
    passed: int = 0
    total: int = 0
    accuracy: float = 0.0
    note: str = ""
    cases: List[dict] = None

class OfficialBenchmarkLoader:
    """Best-effort official dataset runners.

    They run only if the user supplies official files. They do not download or
    fabricate official splits. This prevents benchmark-leakage claims.
    """
    def run_scan_file(self, path: str) -> OfficialLoadReport:
        p=Path(path)
        if not p.exists():
            return OfficialLoadReport("SCAN-official", str(path), False, note="file not found; official dataset not evaluated", cases=[])
        interp=SCANInterpreter(); cases=[]
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line=line.strip()
            if not line or "OUT:" not in line or "IN:" not in line: continue
            m=re.match(r"IN:\s*(.*?)\s*OUT:\s*(.*)$", line)
            if not m: continue
            cmd, exp=m.groups(); pred=interp.predict(cmd)
            cases.append({"input":cmd,"expected":exp.strip(),"predicted":pred,"ok":pred==exp.strip()})
        passed=sum(c["ok"] for c in cases); total=len(cases)
        return OfficialLoadReport("SCAN-official", str(path), True, passed,total,0 if not total else passed/total, "evaluated supplied SCAN-format file", cases[:20])

    def run_babi_file(self, path: str) -> OfficialLoadReport:
        p=Path(path)
        if not p.exists():
            return OfficialLoadReport("bAbI-official", str(path), False, note="file not found; official dataset not evaluated", cases=[])
        w=BabiMicroWorld(); cases=[]
        for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line=re.sub(r"^\d+\s+", "", raw.strip())
            if not line: continue
            if "?" in line and "\t" in line:
                q, ans, *_ = line.split("\t")
                pred=w.answer(q)
                cases.append({"input":q,"expected":ans.strip().lower(),"predicted":pred,"ok":pred==ans.strip().lower()})
            else:
                w.observe(line)
        passed=sum(c["ok"] for c in cases); total=len(cases)
        return OfficialLoadReport("bAbI-official", str(path), True, passed,total,0 if not total else passed/total, "evaluated supplied bAbI-format file", cases[:20])

    def run_clutrr_csv(self, path: str) -> OfficialLoadReport:
        p=Path(path)
        if not p.exists():
            return OfficialLoadReport("CLUTRR-official", str(path), False, note="file not found; official dataset not evaluated", cases=[])
        cases=[]
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            reader=csv.DictReader(f)
            for row in reader:
                story=row.get("story") or row.get("text") or row.get("clean_story") or ""
                query=row.get("query") or row.get("query_text") or ""
                target=(row.get("target") or row.get("answer") or row.get("relation") or "").lower().strip()
                if not story or not target: continue
                kr=KinshipReasoner()
                for sent in re.split(r"[.!?]\s*", story):
                    if sent.strip(): kr.add_fact(sent.strip())
                names=re.findall(r"\b[A-Z][a-z]+\b", query or story)
                if len(names)>=2:
                    pred=kr.relation(names[0], names[-1])
                    cases.append({"input":story+" "+query,"expected":target,"predicted":pred,"ok":pred==target})
        passed=sum(c["ok"] for c in cases); total=len(cases)
        return OfficialLoadReport("CLUTRR-official", str(path), True, passed,total,0 if not total else passed/total, "best-effort supplied CLUTRR CSV evaluation", cases[:20])
