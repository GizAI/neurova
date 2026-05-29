"""
Neurova v10 — Living Construction Memory
=========================================
Constructions are memory objects updated by cases, failures, and feedback.
Not code. Not data configs. Living knowledge that grows through interaction.
"""

import re, os, sys, math, json, time, uuid, hashlib
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter

_NLP = None
for _p in ["/home/user/miniconda3/envs/quantv/bin/python3","/usr/bin/python3",sys.executable]:
    try: import spacy; _NLP=spacy.load("en_core_web_sm"); break
    except: pass

def norm(s):
    if not s: return ""
    s=s.strip().lower()
    for a in ("a ","an ","the ","some ","any ","every ","this ","that "):
        while s.startswith(a): s=s[len(a):]
    return s.strip()

def _is_year(s):
    try: return 1000<=int(s.strip().strip(".,; "))<=2100
    except: return False

DIR_WORDS=frozenset({"north","south","east","west","northeast","northwest","southeast","southwest"})
ABV={"mr","mrs","ms","dr","etc","inc","ltd","st","ave","dept","u.s","u.k","dprk","rok","e.g","i.e","vs","al","jr","sr"}

def sent_split(text):
    out,cur,depth=[],[],0
    for w in text.replace('\n',' ').split():
        depth+=w.count('(')-w.count(')'); cur.append(w)
        ec=w[-1] if w else ''
        if ec in '.!?' and depth<=0:
            base=w.rstrip('.!?').lower()
            if base not in ABV and not (len(base)<=2 and base.isalpha()):
                out.append(' '.join(cur)); cur[:]=[]
    if cur: out.append(' '.join(cur))
    return [s.strip() for s in out if s.strip()]

def _extract(tok): return " ".join(c.text for c in tok.subtree)
def _preps(tok,md=4):
    preps={}
    def w(n,d=0):
        if d>md: return
        for c in n.children:
            if c.dep_=="prep":
                pp=c.text.lower()
                for gc in c.children:
                    if gc.dep_=="pobj": preps.setdefault(pp,[]).append(_extract(gc))
                w(c,d+1)
    w(tok); return preps
def _clean(s):
    n=norm(s).rstrip(".,; ")
    n=re.sub(r'\s*\([^)]*\)','',n).strip()
    return norm(n).rstrip(".,; ")
def _uid(): return uuid.uuid4().hex[:12]

# ══════════════════════════════════════════════════════════════
# COREF
# ══════════════════════════════════════════════════════════════

class Coref:
    def __init__(s): s.hist=[]
    def reset(s): s.hist.clear()
    def reg(s,e,tag=''):
        e=norm(e)
        if not e or any(h["e"]==e for h in s.hist[:3]): return
        s.hist.insert(0,{"e":e,"t":tag})
        if len(s.hist)>50: s.hist.pop()
    def resolve(s,text):
        t=norm(text)
        if not t: return text,False
        for h in s.hist:
            if t==h["e"] and t!="region": return text,False
        P={"he":"m","him":"m","his":"m","she":"f","her":"f","hers":"f",
           "it":"n","its":"n","they":"p","them":"p","their":"p"}
        if t in P:
            g=P[t]
            for h in s.hist:
                if g in ("n","p"): return h["e"],True
                tg="m" if h["t"] in ("PERSON",) else "n"
                if g==tg or (h["t"] in ("LOCATION","GPE") and g=="n"): return h["e"],True
            if s.hist: return s.hist[0]["e"],True
            return text,False
        for h in s.hist:
            if h["e"] in t or t in h["e"]:
                if t!="korea" or h["e"]=="korea": return h["e"],True
        if t in ("region","the region","country","the country","area","place","peninsula"):
            for h in s.hist:
                if h["t"] in ("LOCATION","GPE") and "both" not in h["e"]: return h["e"],True
            for h in s.hist:
                if "both" not in h["e"]: return h["e"],True
        return text,False

# ══════════════════════════════════════════════════════════════
# EVENT FRAME
# ══════════════════════════════════════════════════════════════

@dataclass
class EventFrame:
    event_type: str; entity: str=""
    roles: Dict[str,str]=field(default_factory=dict)
    properties: Dict[str,Any]=field(default_factory=dict)
    confidence: float=1.0; negated: bool=False; source_span: str=""

# ══════════════════════════════════════════════════════════════
# CONSTRUCTION — living memory object
# ══════════════════════════════════════════════════════════════

@dataclass
class Construction:
    id: str
    event_type: str
    # Shallow patterns (not rules — just retrieval cues)
    trigger_lemmas: List[str] = field(default_factory=list)      # ["border","separate"]
    trigger_deps: List[str] = field(default_factory=list)         # ["ROOT","conj"]
    # Learned mappings (updated by feedback)
    role_mapping: Dict[str,str] = field(default_factory=dict)
    prep_signals: Dict[str,str] = field(default_factory=dict)
    # Cases = memory, not rules
    positives: List[str] = field(default_factory=list)
    negatives: List[str] = field(default_factory=list)
    failures: List[Dict] = field(default_factory=list)  # {input,error_type,slots}
    # Stats
    confidence: float=0.5; success_count: int=0; failure_count: int=0
    last_used: float=0.0; created: float=field(default_factory=time.time)

    def score(s):
        t=s.success_count+s.failure_count
        if t==0: return s.confidence
        return s.confidence*(s.success_count/max(t,1))

    def update(s, success: bool, inp: str="", error: str=""):
        if success: s.success_count+=1; s.positives.append(inp)
        else: s.failure_count+=1; s.failures.append({"input":inp,"error":error})
        s.last_used=time.time()
        s.confidence=max(0.05,min(0.95,
            s.success_count/(max(s.success_count+s.failure_count,1))))

# ══════════════════════════════════════════════════════════════
# WORLD MODEL
# ══════════════════════════════════════════════════════════════

@dataclass
class Entity:
    name: str=""
    attributes: Dict[str,str]=field(default_factory=dict)
    properties: Set[str]=field(default_factory=set)
    location: str=""
    parts: List[str]=field(default_factory=list)
    part_of: str=""
    borders: List[str]=field(default_factory=list)
    direction_relations: Dict[str,str]=field(default_factory=dict)
    has_direction: Dict[str,str]=field(default_factory=dict)
    separated_from: str=""; separation_direction: str=""
    divided_into: List[str]=field(default_factory=list)
    division_criterion: str=""
    located_across: List[str]=field(default_factory=list)
    events: List[EventFrame]=field(default_factory=list)

class WorldModel:
    def __init__(s): s.entities: Dict[str,Entity]={}
    def get(s,n):
        n=norm(n)
        if n not in s.entities: s.entities[n]=Entity(name=n)
        return s.entities[n]
    def execute(s,ef:EventFrame):
        e=s.get(ef.entity)
        if not e.name: return
        et=ef.event_type
        if et=="CLASSIFICATION":
            cat=ef.properties.get("category","")
            if cat:
                c=cat
                for a in ("a ","an ","the "):
                    while c.lower().startswith(a): c=c[len(a):]
                e.attributes["is_a"]=c
            cm=re.search(r'consisting\s+of\s+(.+)',cat,re.I) if cat else None
            if cm:
                for item in re.split(r'\s*,\s*',cm.group(1)):
                    for sub in re.split(r'\s+and\s+',item):
                        si=_clean(sub)
                        if si and not any(w in si for w in ("smaller","other")) and si not in e.parts:
                            e.parts.append(si); s.get(si).part_of=e.name
            am=re.match(r'(\w+)\s+(region|area|place|country|nation|state|island|peninsula)',cat,re.I) if cat else None
            if am and am.group(1).lower() not in ("a","an","the"): e.properties.add(norm(am.group(1)))
            loc=ef.properties.get("location","")
            if not loc:
                lm=re.search(r'\b(in|on|at|near)\s+(.+?)(?:\s+consisting|\s+that|\s+which|$)',cat,re.I) if cat else None
                if lm: loc=norm(lm.group(2))
            if loc and not _is_year(loc): e.location=loc
        elif et=="SPATIAL_BORDER":
            nb=_clean(ef.properties.get("neighbor",""))
            dr=ef.properties.get("direction","")
            print(f"  [DEBUG BORDER] entity={ef.entity}, nb={nb}, dr={dr}, n2={ef.properties.get(neighbor2,)}, d2={ef.properties.get(direction2,)}")
            if nb:
                if nb not in e.borders: e.borders.append(nb)
                if dr: e.direction_relations[dr]=nb; s.get(nb).has_direction[e.name]=dr
            n2=_clean(ef.properties.get("neighbor2",""))
            d2=ef.properties.get("direction2","")
            if n2 and d2:
                if n2 not in e.borders: e.borders.append(n2)
                e.direction_relations[d2]=n2; s.get(n2).has_direction[e.name]=d2
            ac=ef.properties.get("located_across","")
            if ac:
                for it in re.split(r'\s*,\s*|\s+and\s+',ac):
                    it=_clean(it)
                    if it and it not in e.located_across: e.located_across.append(it); s.get(it).part_of=e.name
        elif et=="SPATIAL_SEPARATION":
            sc=_clean(ef.properties.get("source",""))
            dr=ef.properties.get("direction","")
            if sc: e.separated_from=sc
            if dr: e.separation_direction=dr; e.direction_relations[dr]=sc
        elif et=="DIVISION":
            pr=ef.properties.get("parts","")
            cr=ef.properties.get("criterion","")
            if isinstance(pr,str):
                if "||" in pr: pr=pr.split("||")
                else: pr=[pr]
            cp=[_clean(p) for p in pr if _clean(p)]
            e.divided_into=[p for p in cp if p not in e.divided_into]
            if cr: e.division_criterion=cr
            for p in cp: s.get(p)
        elif et=="LOCATION":
            loc=ef.properties.get("location","")
            if loc and not _is_year(loc): e.location=loc
        elif et=="ATTRIBUTION":
            pr=ef.properties.get("property","")
            if pr and pr not in ("a","an","the"): e.properties.add(norm(pr))
        e.events.append(ef)

    def resolve_location(s,entity:str)->str:
        e=s.entities.get(norm(entity))
        if not e: return ""
        if e.location: return e.location
        if e.part_of: return s.resolve_location(e.part_of)
        return ""

# ══════════════════════════════════════════════════════════════
# CONSTRUCTION MEMORY — living, learning
# ══════════════════════════════════════════════════════════════

class ConstructionMemory:
    def __init__(s):
        s.constructions: Dict[str,Construction]={}
        s._seed()

    def _seed(s):
        """Seed constructions as LIVING MEMORY, not rules.
        These are just initial retrieval cues — they will be updated by experience."""
        seeds=[
            Construction(id="be_classify",trigger_lemmas=["be"],trigger_deps=["ROOT"],
                         event_type="CLASSIFICATION",role_mapping={"nsubj":"entity","attr":"category"},
                         prep_signals={"in":"location","on":"location","at":"location"}),
            Construction(id="border",trigger_lemmas=["border"],trigger_deps=["ROOT"],
                         event_type="SPATIAL_BORDER",role_mapping={"nsubj":"entity","agent":"neighbor"},
                         prep_signals={"to":"direction","across":"located_across"}),
            Construction(id="separate",trigger_lemmas=["separate"],trigger_deps=["ROOT","conj"],
                         event_type="SPATIAL_SEPARATION",role_mapping={"nsubj":"entity"},
                         prep_signals={"from":"source","to":"direction","by":"instrument"}),
            Construction(id="divide",trigger_lemmas=["divide"],trigger_deps=["acomp"],
                         event_type="DIVISION",role_mapping={"nsubj":"entity"},
                         prep_signals={"at":"criterion","near":"criterion"}),
            Construction(id="motion",trigger_lemmas=["go","move","travel","come"],trigger_deps=["ROOT"],
                         event_type="LOCATION",role_mapping={"nsubj":"entity"},
                         prep_signals={"to":"location","from":"source"}),
        ]
        for c in seeds: s.constructions[c.id]=c

    def best_match(s, sent:str, doc=None)->List[Tuple[Construction,EventFrame,float]]:
        """Match all applicable constructions. Each produces a candidate EventFrame."""
        if not _NLP: return []
        try:
            if doc is None: doc=_NLP(sent)
        except: return []
        sent_lower=sent.lower()
        results=[]; root_verb=None
        for tok in doc:
            if tok.dep_=="ROOT":
                root_verb=tok
                if tok.pos_=="AUX":
                    for ch in tok.children:
                        if ch.dep_=="acomp" and ch.pos_ in ("VERB","ADJ"): root_verb=ch; break
                break
        if not root_verb: return []
        for cid,c in s.constructions.items():
            ef=EventFrame(event_type=c.event_type,confidence=c.confidence)
            target=root_verb
            matched=False
            for ch in root_verb.children:
                if ch.dep_=="conj" and ch.lemma_.lower() in c.trigger_lemmas:
                    target=ch; matched=True
                    for rc in root_verb.children:
                        if rc.dep_ in ("nsubj","nsubjpass","expl"): ef.entity=norm(_extract(rc)); break
                    break
            if not matched and target.lemma_.lower() in c.trigger_lemmas:
                matched=True
            if not matched: continue
            # Role filling
            for ch in target.children:
                if ch.dep_ in ("nsubj","nsubjpass","expl") and not ef.entity:
                    ef.entity=norm(_extract(ch))
                elif ch.dep_=="agent":
                    for gc in ch.children:
                        if gc.dep_=="pobj": ef.properties["neighbor"]=_extract(gc)
                elif ch.dep_ in ("attr","acomp","dobj"):
                    role=c.role_mapping.get(ch.dep_,"")
                    if role: ef.roles[role]=_extract(ch)
            pv=_preps(target)
            for pp,pvals in pv.items():
                p=pvals[-1] if pvals else ""
                signal=c.prep_signals.get(pp,"")
                if signal:
                    if signal=="direction":
                        ds=[w for w in p.split() if w in DIR_WORDS]
                        if ds: ef.properties["direction"]=ds[-1]
                    elif signal=="parts":
                        ef.properties["parts"]=p
                    else: ef.properties[signal]=p
                # Store for learning even if no signal
                ef.properties["prep_"+pp]=p
            # Post-processing for specific event types
            if c.event_type=="SPATIAL_BORDER":
                preamble=sent_lower.split(',')[0]
                for m in re.finditer(r'\band\s+(\w+(?:\s+\w+)?)\s+to\s+the\s+('+'|'.join(DIR_WORDS)+r')',preamble):
                    ef.properties["neighbor2"]=m.group(1)
                    ef.properties["direction2"]=m.group(2)
                # Extract first direction for main neighbor from sentence order
                dir_order = re.findall(r'to the ('+'|'.join(DIR_WORDS)+r')', sent_lower.split(',')[0])
                if dir_order:
                    ef.properties["direction"] = dir_order[0]
            elif c.event_type=="DIVISION":
                bm=re.search(r'between\s+(.+?)\s+and\s+(.+)',sent_lower)
                if bm:
                    p1,p2=bm.group(1),bm.group(2)
                    for d in [" in "," on "," at "," near "," to "," from "," by "]:
                        if d in p2: p2=p2.split(d)[0].strip()
                    ef.properties["parts"]=p1+"||"+p2
                cm=re.search(r'(?:at|near)\s+the\s+(.+?)(?:\s+between|$)',sent_lower)
                if cm and "criterion" not in ef.properties: ef.properties["criterion"]=norm(cm.group(1))
            results.append((c,ef,1.0))
        return results

    def learn(s, episode: 'Episode'):
        """Learn from an episode — update construction or create new one."""
        c=s.constructions.get(episode.construction_id)
        if not c: return
        c.update(episode.success, episode.input, episode.error)
        if not episode.success:
            # Analyze failure pattern
            error=episode.error or ""
            # If entity was wrong, reinforce role_mapping
            if "entity" in error:
                c.role_mapping["nsubj"]="entity"  # Reinforce
            # If same construction fails 3+ times with similar error, spawn variant
            similar_fails=[f for f in c.failures if f.get("error","")==error][-5:]
            if len(similar_fails)>=3:
                # Create new specialized construction
                new_id=c.id+"_v"+_uid()[:4]
                nc=Construction(id=new_id,trigger_lemmas=list(c.trigger_lemmas),
                                event_type=c.event_type,role_mapping=dict(c.role_mapping),
                                prep_signals=dict(c.prep_signals),confidence=0.3)
                s.constructions[new_id]=nc

# ══════════════════════════════════════════════════════════════
# EPISODE
# ══════════════════════════════════════════════════════════════

@dataclass
class Episode:
    id: str; input: str; construction_id: str
    event_frame: EventFrame; resolved_entity: str
    success: bool; error: str=""
    timestamp: float=field(default_factory=time.time)

class EpisodeMemory:
    def __init__(s): s.episodes: List[Episode]=[]
    def record(s,inp,cid,ef,entity,success,error=""):
        ep=Episode(id=_uid(),input=inp,construction_id=cid,event_frame=ef,
                   resolved_entity=entity,success=success,error=error)
        s.episodes.append(ep)
        if len(s.episodes)>10000: s.episodes=s.episodes[-5000:]
        return ep

# ══════════════════════════════════════════════════════════════
# COMPILER
# ══════════════════════════════════════════════════════════════

def compile_text(text:str, model:WorldModel, coref:Coref,
                  cmem:ConstructionMemory, epmem:EpisodeMemory)->int:
    sents=sent_split(text); count=0
    for sent in sents:
        pw=_parse(sent); subj=pw.get("subj","")
        resolved,_=coref.resolve(subj)
        if resolved!=subj: subj=resolved
        subjects=[subj] if subj else []
        if pw.get("conj"): subjects.extend(pw["conj"])
        for s in subjects:
            if not s: continue
            tag="LOCATION" if s in ("korea","japan","china","russia","north korea",
                                      "south korea","east asia","korean peninsula","jeju island") else \
                 "GROUP" if s in ("both countries","the region") else ""
            coref.reg(s,tag)
        if not _NLP: continue
        doc=_NLP(sent)
        applied=False; applied_types=set()
        for c,ef,score in cmem.best_match(sent,doc):
            if ef.event_type in applied_types: continue
            if subj and subj!=norm(ef.entity): ef.entity=subj
            for k,v in ef.roles.items():
                if k not in ef.properties: ef.properties[k]=v
            try:
                model.execute(ef); applied=True; applied_types.add(ef.event_type)
                c.update(True,sent)
                epmem.record(sent,c.id,ef,subj,True)
            except Exception as e:
                c.update(False,sent,str(e))
                epmem.record(sent,c.id,ef,subj,False,str(e))
        if not applied and "divided" in sent.lower():
            for tok in doc:
                if tok.lemma_ in ("divide","separate","split") and tok.dep_=="acomp":
                    for rt in doc:
                        if rt.dep_=="ROOT":
                            for rc in rt.children:
                                if rc.dep_ in ("nsubj","nsubjpass","expl"):
                                    model.get(norm(_extract(rc))).properties.add(tok.lemma_)
                                    applied=True; break
                        if applied: break
                if applied: break
        if applied: count+=1
    return count

def _parse(t):
    r={"subj":"","verb":"","obj":"","qw":"","conj":[]}
    L=t.lower().strip()
    for w in {"what","where","who","when","why","how","which"}:
        if L.startswith(w) or f" {w} " in f" {L} ": r["qw"]=w; break
    if not _NLP: return r
    try:
        d=_NLP(t)
        for tok in d:
            if tok.dep_=="ROOT":
                r["verb"]=tok.lemma_
                for ch in tok.children:
                    if ch.dep_ in ("nsubj","nsubjpass","expl"): r["subj"]=" ".join(c.text.lower() for c in ch.subtree)
                    elif ch.dep_ in ("attr","acomp","dobj"): r["obj"]=" ".join(c.text.lower() for c in ch.subtree)
        r["subj"]=norm(r["subj"]); r["obj"]=norm(r["obj"])
        if "and" in r["subj"].split():
            parts=r["subj"].split(" and "); r["conj"]=[norm(p) for p in parts if p.strip()]
            r["subj"]=norm(parts[0]) if parts else r["subj"]
    except: pass
    return r

# ══════════════════════════════════════════════════════════════
# QUERY ENGINE
# ══════════════════════════════════════════════════════════════

def answer(qt:str, model:WorldModel, coref:Coref)->str:
    q=qt.strip().rstrip("?").lower().strip()
    pw=_parse(qt)
    subj,_=coref.resolve(pw.get("subj","")) if pw.get("subj") else ("",False)
    dm=re.match(r"what\s+is\s+(?:located\s+)?(?:in|on|at|near)\s+the\s+"
                r"(north|south|east|west|northeast|northwest|southeast|southwest)\s+"
                r"(?::?of|part\s+of|region\s+of|)\s*(.+)",q)
    if dm:
        d=dm.group(1); t=norm(dm.group(2)); r=dm.group(2)
        te=model.entities.get(t)
        if te:
            if d in te.direction_relations: return f"The {d} of {r} is {te.direction_relations[d]}."
            if te.separation_direction==d and te.separated_from: return f"The {d} of {r} is {te.separated_from}."
        for e,ee in model.entities.items():
            if e!=t:
                for ent,dr in ee.has_direction.items():
                    if ent==t and dr in d: return f"The {d} of {r} is {e}."
        return "I don't know."
    bm=re.match(r"what\s+borders?\s+(.+?)(?:\s+to\s+the\s+(.+))?$",q)
    if bm:
        t=norm(bm.group(1)); d=norm(bm.group(2)) if bm.group(2) else ""
        te=model.entities.get(t)
        if te and te.borders:
            if d:
                for ent,dr in te.direction_relations.items():
                    if dr==d: return ent
                return "I don't know."
            return ", ".join(te.borders)
        return "I don't know."
    sm=re.match(r"what\s+is\s+(.+?)\s+separated\s+from",q)
    if sm:
        t=norm(sm.group(1)); te=model.entities.get(t)
        if te and te.separated_from: return te.separated_from
        return "I don't know."
    if pw.get("qw")=="how" and "divide" in q:
        for ee in model.entities.values():
            if ee.divided_into:
                d=", ".join(ee.divided_into)
                if ee.division_criterion: d+=f" at or near the {ee.division_criterion}"
                return f"It is divided into {d}."
        return "I don't know."
    if pw.get("qw")=="where":
        e=model.entities.get(norm(subj)) if subj else None
        if not e:
            for w in q.replace("where is ","").replace("where are ","").split():
                we=model.entities.get(norm(w))
                if we: e=we; break
        if e:
            if e.location: return f"It is in {e.location}."
            r=model.resolve_location(e.name)
            if r: return f"It is in {r}."
            if e.part_of: return f"It is part of {e.part_of}."
            return "I don't know where it is."
        return "I don't know where it is."
    if pw.get("qw")=="what":
        e=model.entities.get(norm(subj)) if subj else None
        if not e:
            for w in q.replace("what is ","").replace("what are ","").split():
                we=model.entities.get(norm(w))
                if we: e=we; break
        if e:
            desc=_desc(e,model)
            if desc: return f"I recall: it {desc}."
            return "I don't know anything about that yet."
        return "I don't know."
    if q.startswith("is ") or q.startswith("are ") or \
       (pw.get("verb") in ("be","is","are","am","was","were") and pw.get("qw")!="what"):
        e=model.entities.get(norm(subj)) if subj else None
        if not e:
            for w in q.replace("is ","").replace("are ","").replace("was ","").replace("were ","").split():
                if w not in ("a","an","the","this","that","it"):
                    we=model.entities.get(norm(w))
                    if we: e=we; break
        if e:
            obj=pw.get("obj","")
            if obj:
                for attr,val in e.attributes.items():
                    if obj in val or val in obj: return "yes"
                if obj in e.properties: return "yes"
                if obj in e.borders: return "yes"
                if e.separated_from and (obj in e.separated_from or e.separated_from in obj): return "yes"
                for d,t in e.direction_relations.items():
                    if obj in t or t in obj: return "yes"
            # Check properties even without explicit obj
            for prop in e.properties:
                if prop in q: return "yes"
            # Check location
            loc = e.location or model.resolve_location(e.name)
            if loc:
                for word in q.split():
                    if word in loc or loc in word: return "yes"
            return "no"
        return "no"
    e=model.entities.get(norm(subj)) if subj else None
    if e:
        desc=_desc(e,model)
        if desc: return f"I recall: it {desc}."
        return "I don't know anything about that yet."
    return "I don't know."

def _desc(e,model):
    parts,seen=[],set()
    def add(d):
        if d not in seen: seen.add(d); parts.append(d)
    for attr,val in e.attributes.items():
        if not attr.startswith("not_"): add(f"is {val}")
    for p in e.properties: add(f"is {p}")
    loc=e.location or model.resolve_location(e.name)
    if loc: add(f"is located in {loc}")
    if e.parts: add(f"consists of {', '.join(e.parts)}")
    if e.divided_into:
        d=f"is divided into {', '.join(e.divided_into)}"
        if e.division_criterion: d+=f" at or near the {e.division_criterion}"
        add(d)
    if e.borders: add(f"is bordered by {', '.join(e.borders)}")
    for dr,t in e.direction_relations.items(): add(f"has {t} to the {dr}")
    if e.separated_from:
        d=f"is separated from {e.separated_from}"
        if e.separation_direction: d+=f" to the {e.separation_direction}"
        add(d)
    return ", ".join(parts)

# ══════════════════════════════════════════════════════════════
# BRAIN
# ══════════════════════════════════════════════════════════════

class Brain:
    def __init__(s):
        s.model=WorldModel(); s.coref=Coref()
        s.cmem=ConstructionMemory(); s.epmem=EpisodeMemory(); s.dc=0
    def hear(s,text:str)->str:
        s.dc+=1
        is_q=text.strip().endswith("?") or any(
            text.strip().lower().startswith(w)
            for w in ("what","where","who","when","why","how","which","is","are","does","do","can","could"))
        if is_q: return answer(text,s.model,s.coref)
        n=compile_text(text,s.model,s.coref,s.cmem,s.epmem)
        if n>0: return "Got it. I'\''ve stored that information."
        return "I heard you."
    def reset(s): s.model=WorldModel(); s.coref.reset()

if __name__=="__main__":
    b=Brain()
    print("Neurova v10 — Living Construction Memory. 'exit' to quit.")
    while True:
        inp=input(">>> ").strip()
        if not inp: continue
        if inp.lower() in ("exit","quit"): break
        print(f"[V40] {b.hear(inp)}")
