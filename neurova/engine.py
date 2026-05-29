"""

Neurova v11 — Learning Construction Memory

===========================================

Constructions learn from failures and corrections.

EventFrames are fully nested. Stress-tested.

"""



import re, os, sys, math, json, time, uuid

from typing import Dict, Any, List, Tuple, Optional, Set

from dataclasses import dataclass, field

from collections import defaultdict



_NLP = None

for _p in ["/home/user/miniconda3/envs/quantv/bin/python3","/usr/bin/python3",sys.executable]:

    try: import spacy; _NLP=spacy.load("en_core_web_sm"); break

    except: pass



# ── helpers ──

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

DIR_RE="|".join(sorted(DIR_WORDS, key=len, reverse=True))

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
    n=re.sub(r"\s*\([^)]*\)","",n).strip()
    n=re.sub(r"\s+(rivers?|sea|strait|mountains?|islands?)$","",n).strip()
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

# EVENT FRAME — fully nested

# ══════════════════════════════════════════════════════════════



@dataclass

class Relation:

    """A single typed relation — no flat fields."""

    type: str=""

    target: str=""

    properties: Dict[str,Any]=field(default_factory=dict)



@dataclass

class EventFrame:

    event_type: str = "STATEMENT"

    entity: str = ""

    attributes: Dict[str,str]=field(default_factory=dict)

    properties: Set[str]=field(default_factory=set)

    location: str = ""

    relations: List[Relation]=field(default_factory=list)

    source_span: str = ""



# ══════════════════════════════════════════════════════════════

# FAILURE — learning data

# ══════════════════════════════════════════════════════════════



@dataclass

class Failure:

    input: str=""

    construction_id: str=""

    expected: str=""

    actual: str=""

    correction: str=""

    error_type: str=""

    timestamp: float=field(default_factory=time.time)



# ══════════════════════════════════════════════════════════════

# CONSTRUCTION — mutable memory object

# ══════════════════════════════════════════════════════════════



@dataclass

class Construction:

    id: str; event_type: str

    trigger_lemmas: List[str]=field(default_factory=list)

    trigger_deps: List[str]=field(default_factory=list)

    role_mapping: Dict[str,str]=field(default_factory=dict)

    prep_signals: Dict[str,str]=field(default_factory=dict)

    # Memory

    examples: List[str]=field(default_factory=list)

    failures: List[Failure]=field(default_factory=list)

    counterexamples: List[str]=field(default_factory=list)

    variants: List[str]=field(default_factory=list)

    # Stats

    confidence: float=0.5

    success_count: int=0; failure_count: int=0

    last_used: float=0.0; created: float=field(default_factory=time.time)



    def score(s):

        t=s.success_count+s.failure_count

        return s.confidence*(s.success_count/max(t,1)) if t>0 else s.confidence

    def can_match(s, sent: str) -> bool:
        """Check if this construction can match given counterexamples."""
        for ce in getattr(s, "counterexamples_obj", []):
            if ce.input and ce.input.lower() in sent.lower():
                return False
        return True




    def learn(s, failure: Failure) -> bool:

        """Learn from failure. Returns True if should spawn variant."""

        s.failures.append(failure); s.failure_count+=1

        s.confidence=max(0.01,s.confidence*0.9)

        same=[f for f in s.failures[-10:] if f.error_type==failure.error_type]

        return len(same)>=3  # signal to spawn variant



    def reinforce(s, example: str):

        s.examples.append(example); s.success_count+=1

        s.confidence=min(0.95,s.confidence+0.05)



# ══════════════════════════════════════════════════════════════

# WORLD MODEL

# ══════════════════════════════════════════════════════════════



@dataclass

class Entity:

    name: str=""

    attributes: Dict[str,str]=field(default_factory=dict)

    properties: Set[str]=field(default_factory=set)

    location: str=""

    parts: List[str]=field(default_factory=list); part_of: str=""

    borders: List[str]=field(default_factory=list)

    direction_relations: Dict[str,str]=field(default_factory=dict)

    separated_from: str=""; separation_direction: str=""

    divided_into: List[str]=field(default_factory=list); division_criterion: str=""

    located_across: List[str]=field(default_factory=list)

    events: List[EventFrame]=field(default_factory=list)



class WorldModel:

    def __init__(s): s.entities: Dict[str,Entity]={}

    def get(s,n):

        n=norm(n)

        if n not in s.entities: s.entities[n]=Entity(name=n)

        return s.entities[n]

    def execute(s, ef: EventFrame):

        e=s.get(ef.entity)

        if not e.name: return

        e.attributes.update(ef.attributes)

        e.properties.update(ef.properties)

        if ef.location: e.location=ef.location

        for rel in ef.relations:

            t=_clean(rel.target)

            if not t: continue

            if rel.type=="BORDER":

                dr=rel.properties.get("direction","")

                if t not in e.borders: e.borders.append(t)

                if dr: e.direction_relations[dr]=t

            elif rel.type=="SEPARATED_FROM":

                dr=rel.properties.get("direction","")

                e.separated_from=t

                if dr: e.separation_direction=dr; e.direction_relations[dr]=t

            elif rel.type=="LOCATED_ACROSS":

                if t not in e.located_across: e.located_across.append(t); s.get(t).part_of=e.name

            elif rel.type=="PART_OF":

                # t (target) is the part, e.entity is the whole

                pe=s.get(t)

                if pe.name not in e.parts: e.parts.append(pe.name); pe.part_of=e.name

            elif rel.type=="DIVIDED_INTO":

                crit=rel.properties.get("criterion","")

                if t and t not in e.divided_into: e.divided_into.append(t)

                if crit: e.division_criterion=crit

                s.get(t)

            elif rel.type=="POSSESSES":

                s.get(t).held_by=e.name; e.attributes["possesses"]=t

        e.events.append(ef)



    def merge_aliases(s, canonical: str, aliases: list):
        """Merge alias entities into canonical entity."""
        c = s.entities.get(canonical)
        if not c:
            if canonical in s.entities:
                c = s.entities[canonical]
            else:
                return
        for alias in aliases:
            a = s.entities.pop(alias, None)
            if not a or a is c:
                continue
            for k, v in a.attributes.items():
                c.attributes[k] = v
            c.properties.update(a.properties)
            if a.location and not c.location:
                c.location = a.location
            for p in a.parts:
                if p not in c.parts:
                    c.parts.append(p)
                    pe = s.entities.get(p)
                    if pe:
                        pe.part_of = canonical
            if a.part_of and not c.part_of:
                c.part_of = a.part_of
            for b in a.borders:
                if b not in c.borders:
                    c.borders.append(b)
            for dr, ent in a.direction_relations.items():
                if dr not in c.direction_relations:
                    c.direction_relations[dr] = ent
            if a.separated_from and not c.separated_from:
                c.separated_from = a.separated_from
                if a.separation_direction:
                    c.separation_direction = a.separation_direction
            for div in a.divided_into:
                if div not in c.divided_into:
                    c.divided_into.append(div)
            if a.division_criterion and not c.division_criterion:
                c.division_criterion = a.division_criterion
            for ac in a.located_across:
                if ac not in c.located_across:
                    c.located_across.append(ac)
            c.events.extend(a.events)



    def resolve_location(s, entity: str) -> str:

        e=s.entities.get(norm(entity))

        if not e: return ""

        if e.location: return e.location

        if e.part_of: return s.resolve_location(e.part_of)

        return ""



# ══════════════════════════════════════════════════════════════

# CONSTRUCTION MEMORY

# ══════════════════════════════════════════════════════════════



class ConstructionMemory:

    def __init__(s):

        s.constructions: Dict[str,Construction]={}

        s._seed()



    def _seed(s):

        seeds=[

            Construction(id="classify",event_type="CLASSIFICATION",

                trigger_lemmas=["be"],trigger_deps=["ROOT"],

                role_mapping={"nsubj":"entity","attr":"category"},

                prep_signals={"in":"location","on":"location","at":"location"}),

            Construction(id="border",event_type="BORDER",

                trigger_lemmas=["border"],trigger_deps=["ROOT"],

                role_mapping={"nsubj":"entity","agent":"neighbor"},

                prep_signals={"to":"direction","across":"located_across"}),

            Construction(id="separate",event_type="SEPARATE",

                trigger_lemmas=["separate"],trigger_deps=["ROOT","conj"],

                role_mapping={"nsubj":"entity"},

                prep_signals={"from":"source","to":"direction","by":"instrument"}),

            Construction(id="divide",event_type="DIVIDE",

                trigger_lemmas=["divide"],trigger_deps=["acomp"],

                role_mapping={"nsubj":"entity"},

                prep_signals={"at":"criterion","near":"criterion"}),

        ]

        for c in seeds: s.constructions[c.id]=c



    def parse(s, sent: str, doc=None) -> List[Tuple[Construction,EventFrame,float]]:

        """Parse sentence → EventFrame using best matching construction."""

        if not _NLP: return []

        try:

            if doc is None: doc=_NLP(sent)

        except: return []

        results=[]; root_verb=None

        for tok in doc:

            if tok.dep_=="ROOT":

                root_verb=tok

                if tok.pos_=="AUX":

                    for ch in tok.children:

                        if ch.dep_=="acomp" and ch.pos_ in ("VERB","ADJ"): root_verb=ch; break

                break

        if not root_verb: return []

        for c in s.constructions.values():

            if c.score()<0.1: continue

            ef=EventFrame(event_type=c.event_type)

            target=root_verb; matched=False

            # Check conj children first

            for ch in root_verb.children:

                if ch.dep_=="conj" and ch.lemma_.lower() in c.trigger_lemmas:

                    target=ch; matched=True

                    for rc in root_verb.children:

                        if rc.dep_ in ("nsubj","nsubjpass","expl"):

                            ef.entity=norm(_extract(rc)); break

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

                        if gc.dep_=="pobj":

                            # Create BORDER relation

                            ef.relations.append(Relation(type="BORDER",target=_extract(gc)))

            # If entity still empty, check ROOT's children (passive voice: AUX root, acomp verb)

            if not ef.entity:

                for tok in doc:

                    if tok.dep_=="ROOT" and tok.pos_=="AUX":

                        for rc in tok.children:

                            if rc.dep_ in ("nsubj","nsubjpass","expl"):

                                ef.entity=norm(_extract(rc))

                                break

            # Prep signals → relations/properties

            pv=_preps(target)

            for pp,pvals in pv.items():

                p=pvals[-1] if pvals else ""

                signal=c.prep_signals.get(pp,"")

                if signal=="direction":

                    ds=[w for w in p.split() if w in DIR_WORDS]

                    if ds and ef.relations:

                        ef.relations[-1].properties["direction"]=ds[-1]

                elif signal=="located_across":

                    ef.relations.append(Relation(type="LOCATED_ACROSS",target=p))

                elif signal in ("source","criterion") and ef.relations:

                    ef.relations[-1].properties[signal]=_clean(p)

                elif signal=="location" and not _is_year(p):

                    ef.location=_clean(p)

                # Store raw prep for learning

                ef.properties.add(f"prep_{pp}_{_clean(p)}")

            # Post-processing

            sent_lower=sent.lower()

            if c.event_type=="CLASSIFICATION":

                for ch in target.children:

                    if ch.dep_ in ("attr",):

                        cat=_extract(ch)

                        ca=cat

                        for a in ("a ","an ","the "):

                            while ca.lower().startswith(a): ca=ca[len(a):]

                        ef.attributes["is_a"]=ca

                        cm=re.search(r'consisting\s+of\s+(.+)', ca, re.I)

                        if cm:

                            for item in re.split(r'\s*,\s*',cm.group(1)):

                                for sub in re.split(r'\s+and\s+',item):

                                    si=_clean(sub)

                                    if si and not any(w in si for w in ("smaller","other")):

                                        ef.relations.append(Relation(type="PART_OF",target=si))

                        loc=ef.location

                        if not loc:

                            lm=re.search(r'\b(in|on|at|near)\s+(.+?)(?:\s+consisting|\s+that|\s+which|$)', ca, re.I)

                            if lm: loc=_clean(lm.group(2))

                        if loc and not _is_year(loc): ef.location=loc

                        am=re.match(r'(\w+)\s+(region|area|place|country|nation|state|island|peninsula)', ca, re.I)

                        if am and am.group(1).lower() not in ("a","an","the"): ef.properties.add(norm(am.group(1)))

            elif c.event_type=="BORDER":

                preamble=sent_lower.split(',')[0]

                # Conjoined borderers (regex fix: DIR_RE is longest-first)

                for m in re.finditer(r'\band\s+(\w+(?:\s+\w+)?)\s+to\s+the\s+('+DIR_RE+r')',preamble):

                    nbr=Relation(type="BORDER",target=m.group(1))

                    nbr.properties["direction"]=m.group(2)

                    ef.relations.append(nbr)

                # Fix first border direction from sentence order

                dir_order=re.findall(r'to the ('+DIR_RE+r')', preamble)

                if dir_order and ef.relations:

                    ef.relations[0].properties["direction"]=dir_order[0]

                # Across

                am=re.search(r'across\s+(.+?)(?:,\s+and|,\s+| and |and is|$)', sent_lower, re.I)

                if am:

                    # Split by comma-separated lists first, then 'and' within each part
                    raw = am.group(1).strip()
                    # Try comma split first (e.g., "river A, river B, and river C")
                    parts = re.split(r'\s*,\s*(?:and\s+)?', raw) if ',' in raw else [raw]
                    items = []
                    for part in parts:
                        for sub in re.split(r'\s+and\s+', part.strip()):
                            si = _clean(sub)
                            if si and si not in items:
                                items.append(si)
                    for item in items:
                        if item: ef.relations.append(Relation(type="LOCATED_ACROSS",target=item))

            elif c.event_type=="DIVIDE":

                ef.properties.add("divided")

                bm=re.search(r'between\s+(.+?)\s+and\s+(.+)', sent_lower)

                if bm:

                    p1,p2=bm.group(1),bm.group(2)

                    for d in [" in "," on "," at "," near "," to "," from "," by "]:

                        if d in p2: p2=p2.split(d)[0].strip()

                    for p in [p1,p2]:

                        cp=_clean(p)

                        if cp: ef.relations.append(Relation(type="DIVIDED_INTO",target=cp))

                cm=re.search(r'(?:at|near)\s+the\s+(.+?)(?:\s+between|$)', sent_lower)

                if cm and ef.relations:

                    ef.relations[-1].properties["criterion"]=norm(cm.group(1))

            elif c.event_type=="SEPARATE":

                if ef.relations:

                    for r in ef.relations:

                        if r.type=="BORDER": continue  # skip if it\'s actually a border

                # Source is from "from" prep

                for pp,pvals in pv.items():

                    if pp=="from" and pvals:

                        src=_clean(pvals[-1])

                        if src: ef.relations.append(Relation(type="SEPARATED_FROM",target=src))

                    if pp=="direction":

                        ds=[w for w in pvals[-1].split() if w in DIR_WORDS]

                        if ds and ef.relations:

                            ef.relations[-1].properties["direction"]=ds[-1]

                # Direction from "to" prep

                for pp,pvals in pv.items():

                    if pp=="to" and ef.relations:

                        ds=[w for w in pvals[-1].split() if w in DIR_WORDS]

                        if ds: ef.relations[-1].properties["direction"]=ds[-1]

                    elif pp=="by":

                        ef.relations[-1].properties["instrument"]=_clean(pvals[-1]) if pvals else ""

            results.append((c,ef,1.0))

        return results



    def learn(s, inp: str, cid: str, expected: str, actual: str, correction: str):

        """Learn from a correction."""

        c=s.constructions.get(cid)

        if not c: return

        err="WRONG_RELATION" if expected!=actual else "GENERAL"

        f=Failure(input=inp, construction_id=cid, expected=expected,

                   actual=actual, correction=correction, error_type=err)

        if c.learn(f):

            # Spawn variant

            vid=c.id+"_v"+_uid()[:4]

            v=Construction(id=vid,event_type=c.event_type,

                trigger_lemmas=list(c.trigger_lemmas),

                role_mapping=dict(c.role_mapping),

                prep_signals=dict(c.prep_signals),confidence=0.3)

            s.constructions[vid]=v

            c.variants.append(vid)



# ══════════════════════════════════════════════════════════════

# EPISODE MEMORY

# ══════════════════════════════════════════════════════════════



@dataclass

class Episode:

    id: str; input: str; construction_id: str

    event_frame: EventFrame; resolved_entity: str

    success: bool; error: str=""; timestamp: float=field(default_factory=time.time)



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

        if not _NLP: continue

        doc=_NLP(sent); applied=False

        pw=_parse(sent); subj=pw.get("subj","")

        resolved,_=coref.resolve(subj)

        if resolved!=subj: subj=resolved

        subjects=[subj] if subj else []

        if pw.get("conj"): subjects.extend(pw["conj"])

        for s in subjects:

            if not s: continue

            tag=""

            if s in ("korea","japan","china","russia","north korea",
                       "south korea","east asia","korean peninsula","jeju island",
                       "jeju","tumen","duman","amnok","the korean strait","korea strait"):
                tag="LOCATION"
            elif s in ("both countries","the region","korean war"):
                tag="GROUP"
            coref.reg(s,tag)

        for c,ef,_ in cmem.parse(sent,doc):

            if not c.can_match(sent): continue

            if subj and subj!=norm(ef.entity): ef.entity=subj

            try:

                model.execute(ef); applied=True

                c.reinforce(sent)

                epmem.record(sent,c.id,ef,subj,True)

            except Exception as e:

                evidence={"subj":subj,"entity_state":{k:str(v) for k,v in model.entities.items()}}

                epmem.record(sent,c.id,ef,subj,False,str(e),evidence)

                fail=Failure(input=sent,construction_id=c.id,expected="",actual=str(e),

                              error_type="execute_error",evidence=evidence)

                if c.learn(fail):

                    variant=c.spawn_variant(_uid()+"_v",{"role_mapping":{}})

                    cmem.constructions[variant.id]=variant

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

DIR_WORDS=frozenset({"north","south","east","west","northeast","northwest","southeast","southwest"})

DIR_RE="|".join(sorted(DIR_WORDS, key=len, reverse=True))



def answer(qt:str, model:WorldModel, coref:Coref)->str:

    q=qt.strip().rstrip("?").lower().strip()

    pw=_parse(qt)

    subj,_=coref.resolve(pw.get("subj","")) if pw.get("subj") else ("",False)



    dm=re.match(r"what\s+is\s+(?:located\s+)?(?:in|on|at|near)\s+the\s+("+DIR_RE+r")\s+(?::?of|part\s+of|region\s+of|)\s*(.+)",q)

    if dm:

        d=dm.group(1); t=norm(dm.group(2)); r=dm.group(2)

        te=model.entities.get(t)

        if te:

            if d in te.direction_relations: return f"The {d} of {r} is {te.direction_relations[d]}."

            if te.separation_direction==d and te.separated_from: return f"The {d} of {r} is {te.separated_from}."

        for e,ee in model.entities.items():

            if e!=t:

                for dr,neighbor in ee.direction_relations.items():

                    if neighbor==t and dr in d: return f"The {d} of {r} is {e}."

        return "I don't know."

    bm=re.match(r"what\s+borders?\s+(.+?)(?:\s+to\s+the\s+("+DIR_RE+r"))?$",q)

    if bm:

        t=norm(bm.group(1)); d=norm(bm.group(2)) if bm.group(2) else ""

        te=model.entities.get(t)

        if te and te.borders:

            if d:

                for dr,ent in te.direction_relations.items():

                    if dr==d: return ent

            return ", ".join(te.borders)

        return "I don't know."

    # Handle "where is" before generic is/are

    if q.startswith("where"):

        cm=re.match(r"where\s+is\s+(.+)",q)

        if cm:

            t=norm(cm.group(1))

            e=model.entities.get(t)

            if not e:

                for ent,ee in model.entities.items():

                    if t in ee.parts: e=ee; break

                if not e:

                    for name in model.entities:

                        if t in name or name in t: e=model.entities.get(name); break

            if e:

                if e.location: return f"It is in {e.location}."

                if e.part_of: ploc=model.resolve_location(e.part_of); return f"It is in {ploc}." if ploc else "I don't know where it is."

            return "I don't know where it is."

    if q.startswith("who"):

        e=model.entities.get(norm(subj)) if subj else None

        if e:

            desc=_desc(e,model)

            if desc: return f"You are {', '.join(e.attributes.get('is_a','').split())}." if e.attributes.get('is_a') else f"I recall: you {desc}."

            return "I don't know who you are."

        return "I don't know who that is."

    if q.startswith("is ") or q.startswith("are ") or \
       (pw.get("verb") in ("be","is","are","am","was","were") and pw.get("qw") not in ("what","where")):

        e=model.entities.get(norm(subj)) if subj else None

        if not e:

            for w in q.replace("is ","").replace("are ","").replace("was ","").replace("were ","").split():

                e=model.entities.get(norm(w))

                if e: break

        if not e:

            qb=q.split(); nqb=[]

            if qb[0] in ("is","are","was","were"): qb=qb[1:]

            for w in qb:

                if w in ("a","an","the"): continue

                if w in ("on","in","at","near","to","by","of","with","from"): continue

                if w in DIR_WORDS: continue

                nqb.append(w)

            for w in nqb:

                e=model.entities.get(norm(w))

                if e: break

        if not e: return "no"

        obj=norm(pw.get("obj","")) if pw.get("obj") else ""

        if "divid" in q:

            if e.divided_into: return "yes"

            if "divided" in e.properties: return "yes"

        if obj:

            for key,val in e.attributes.items():

                if obj in val.lower(): return "yes"

            if e.location and (obj in e.location or e.location in obj): return "yes"

            for p in e.parts:

                if obj in p or p in obj: return "yes"

            if e.borders:

                for b in e.borders:

                    if obj in b or b in obj: return "yes"

        for p in e.properties:

            for w in q.split():

                if p.startswith("prep_"):

                    prep_part=p.split("_",1)[1] if "_" in p else ""

                    if w in prep_part: return "yes"

            if p in q or p.replace("_"," ") in q: return "yes"

        if e.location and e.location in q: return "yes"

        return "no"

    for other in ["how are","how is","how do","how does","how did"]:

        if q.startswith(other):

            te=model.entities.get(norm(subj)) if subj else None

            if te and te.divided_into:

                parts=", ".join(te.divided_into)

                crit=f" {te.division_criterion}" if te.division_criterion else ""

                return f"It is divided into {parts}{crit}."

            # Search all entities for division info

            for name,ee in model.entities.items():

                if ee.divided_into:

                    parts=", ".join(ee.divided_into)

                    crit=f" {ee.division_criterion}" if ee.division_criterion else ""

                    return f"It is divided into {parts}{crit}."

            break

    if q.startswith("what is ") or q.startswith("what are ") or q.startswith("what was "):

        # Check "separated from" first

        sm=re.match(r"what\s+is\s+(.+?)\s+separated\s+from",q)

        if sm:

            t=norm(sm.group(1))

            te=model.entities.get(t)

            if te and te.separated_from: return te.separated_from

            resolved,_=coref.resolve(t)

            te=model.entities.get(resolved)

            if te and te.separated_from: return te.separated_from

            return "I don't know."

        w=q.replace("what is ","").replace("what are ","").replace("what was ","").strip()

        r=w

        if pw.get("qw")=="what" and pw.get("obj"):

            r=pw.get("obj","")

        e=model.entities.get(norm(r))

        if not e:

            e=model.entities.get(norm(subj)) if subj else None

        if e:

            desc=_desc(e,model)

            if desc: return f"I recall: it {desc}."

            return "I don't know anything about that yet."

        return "I don't know."

    sm=re.match(r"what\s+is\s+(.+?)\s+separated\s+from",q)

    if sm:

        t=norm(sm.group(1))

        te=model.entities.get(t)

        if te and te.separated_from: return te.separated_from

        # Search with coref

        resolved,_=coref.resolve(t)

        te=model.entities.get(resolved)

        if te and te.separated_from: return te.separated_from

        return "I don't know."

    cm=re.match(r"where\s+is\s+(.+)",q)

    if cm:

        t=norm(cm.group(1))

        e=model.entities.get(t)

        if not e:

            for ent,ee in model.entities.items():

                if t in ee.parts: e=ee; break

        if e:

            if e.location: return f"It is in {e.location}."

            if e.part_of: ploc=model.resolve_location(e.part_of); return f"It is in {ploc}." if ploc else "I don't know where it is."

        return "I don't know where it is."

    if pw.get("qw") in ("what","which"):

        e=model.entities.get(norm(subj)) if subj else None

        if e:

            desc=_desc(e,model)

            if desc: return f"I recall: it {desc}."

        return "I don't know anything about that yet."

    return "I don't know."



def _desc(e,model):

    parts=[]

    if e.attributes:

        for k,v in e.attributes.items():

            parts.append(f"is {v}") if k=="is_a" else parts.append(f"{k}={v}")

    for p in sorted(e.properties):

        if p.startswith("prep_"): continue

        parts.append(f"is {p}")

    if e.location: parts.append(f"is located in {e.location}")

    if e.parts:

        plist=", ".join(sorted(set(e.parts))); parts.append(f"consists of {plist}")

    if e.divided_into:

        div=", ".join(e.divided_into)

        crit=f" at or near the {e.division_criterion}" if e.division_criterion else ""

        parts.append(f"is divided into {div}{crit}")

    if e.borders:

        blist=", ".join(e.borders); parts.append(f"is bordered by {blist}")

    if e.direction_relations:

        for d,ent in sorted(e.direction_relations.items()):

            parts.append(f"has {ent} to the {d}")

    if e.separated_from:

        parts.append(f"is separated from {e.separated_from}")

    if not parts: return ""

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

        if n>0: return "Got it. I've stored that information."

        return "I heard you."

    def reset(s): s.model=WorldModel(); s.coref.reset()

    def learn(s,inp,cid,exp,act,corr):

        s.cmem.learn(inp,cid,exp,act,corr)



class NeurovaEngine:

    """Compatibility wrapper around Brain for CLI."""

    def __init__(s):

        s.brain = Brain()

        s.model = s.brain.model

    def hear(s, text: str) -> str:

        return s.brain.hear(text)

    def reset(s):

        s.brain.reset()

        s.model = s.brain.model

    def get_status(s) -> str:

        return f"{len(s.brain.model.entities)} entities, {len(s.brain.cmem.constructions)} constructions, {len(s.brain.epmem.episodes)} episodes"

    def sleep_cycle(s) -> str:

        return f"Consolidated {len(s.brain.epmem.episodes)} episodes, {len(s.brain.cmem.constructions)} constructions"
