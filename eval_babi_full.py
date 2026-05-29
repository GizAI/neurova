"""
bAbI 20 Tasks — Full Evaluation Harness
Uses the EntityKnowledgeGraph engine for genuine reasoning.
No hardcoded task-specific templates.
"""

import re
import sys
import time
from neurova.architecture.entity_kg import NeuroSymbolicEngine, norm, split_sentences, parse_sentence, LOCATIVE_VERBS, POSSESSION_VERBS, DROP_VERBS, GIVE_VERB, BE_VERBS


TASKS = [
    ("qa1_single-supporting-fact", "Single Supporting Fact"),
    ("qa2_two-supporting-facts", "Two Supporting Facts"),
    ("qa3_three-supporting-facts", "Three Supporting Facts"),
    ("qa4_two-arg-relations", "Two Arg Relations"),
    ("qa5_three-arg-relations", "Three Arg Relations"),
    ("qa6_yes-no-questions", "Yes/No Questions"),
    ("qa7_counting", "Counting"),
    ("qa8_lists-sets", "Lists/Sets"),
    ("qa9_simple-negation", "Simple Negation"),
    ("qa10_indefinite-knowledge", "Indefinite Knowledge"),
    ("qa11_basic-coreference", "Basic Coreference"),
    ("qa12_conjunction", "Conjunction"),
    ("qa13_compound-coreference", "Compound Coreference"),
    ("qa14_time-reasoning", "Time Reasoning"),
    ("qa15_basic-deduction", "Basic Deduction"),
    ("qa16_basic-induction", "Basic Induction"),
    ("qa17_positional-reasoning", "Positional Reasoning"),
    ("qa18_size-reasoning", "Size Reasoning"),
    ("qa19_path-finding", "Path Finding"),
    ("qa20_agents-motivations", "Agents Motivations"),
]


def load_babi_file(path: str) -> list:
    """Load a bAbI file and return structured entries."""
    entries = []
    current_story = []
    
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_story:
                    entries.append(current_story)
                    current_story = []
                continue
            
            m = re.match(r"^(\d+)\s+(.*)", line)
            if not m:
                continue
            
            lineno = int(m.group(1))
            content = m.group(2)
            
            if "\t" in content:
                parts = content.split("\t")
                question = parts[0].strip()
                answer = parts[1].strip()
                support = parts[2].strip() if len(parts) > 2 else ""
                current_story.append({
                    "type": "question",
                    "question": question,
                    "answer": answer,
                    "support": support,
                    "lineno": lineno,
                })
            else:
                current_story.append({
                    "type": "statement",
                    "text": content,
                    "lineno": lineno,
                })
    
    if current_story:
        entries.append(current_story)
    
    return entries


class BabiDetailedEngine(NeuroSymbolicEngine):
    """Extended engine for bAbI with detailed question handling."""
    
    def hear(self, text: str) -> str:
        """Override to handle bAbI-specific query patterns precisely."""
        parsed = parse_sentence(text)
        
        if parsed["is_question"]:
            return self._answer_babi_question(text)
        else:
            facts = self.kg.process_statement(text)
            return "ok"
    
    def _answer_babi_question(self, text: str) -> str:
        """Answer a bAbI question with exact format matching."""
        raw = text.strip().rstrip("?").lower()
        parsed = parse_sentence(text)
        subj = self.kg.coref.resolve(parsed["subject"])
        verb = parsed["verb"]
        obj = parsed["object"]
        qword = parsed["qword"]
        
        # ── WHERE questions ──
        if qword == "where":
            loc = self.kg.get_location(subj)
            if loc:
                return loc
            return ""
        
        # ── WHO questions ──
        if qword == "who":
            if obj:
                for f in self.kg.facts:
                    if f.rel == "be" and f.obj and (obj in f.obj or f.obj in obj) and not f.neg:
                        return f.subj
            return ""
        
        # ── WHAT questions ──
        if qword == "what":
            # "What is Y?" or "What does X verb?" or "What did X give to Y?"
            if subj:
                # For "What did X give to Y?" - find objects given FROM subj TO others
                given_objs = set()
                for f in self.kg.facts:
                    if f.subj == subj and f.rel in GIVE_VERB and f.obj:
                        given_objs.add(f.obj)
                    # Object with recipient
                    if f.rel == "with" and f.obj == subj:
                        pass  # Object is with subj
                if given_objs:
                    return " ".join(given_objs)
                
                # What is X carrying / holding?
                possessions = set()
                for f in self.kg.facts:
                    if f.obj == subj and f.rel == "with" and not f.neg:
                        possessions.add(f.subj)
                if possessions:
                    return " ".join(possessions)
                
                # What is X? → description
                parents = self.kg.get_parents(subj)
                if parents:
                    return parents[-1]
            
            return ""
        
        # ── IS/ARE questions (yes/no) ──
        if raw.startswith("is ") or raw.startswith("are ") or verb in ("be",):
            found, neg = self.kg.verify(subj, obj, verb)
            if neg:
                return "no"
            if found:
                return "yes"
            if obj and self.kg.is_a(subj, obj):
                return "yes"
            return "no"
        
        # ── DOES/DO/CAN questions ──
        if verb in ("do", "does", "did", "can", "could", "will", "would"):
            found, neg = self.kg.verify(subj, obj, verb)
            if found and not neg:
                return "yes"
            if neg:
                return "no"
            # Handle "Does X have Y?" → check possession
            if verb in ("do", "does", "did") and obj:
                for f in self.kg.facts:
                    if f.subj == norm(obj) and f.rel == "with" and f.obj == subj and not f.neg:
                        return "yes"
                    if f.subj == obj and f.rel == "with" and f.obj == subj and not f.neg:
                        return "yes"
            # Handle verb property check
            for f in self.kg.facts:
                if f.subj == subj and f.rel and not f.neg:
                    if obj and f.obj and (obj in f.obj or f.obj in obj):
                        return "yes"
            # Inheritance check
            ancestors = self.kg.get_ancestors(subj)
            for a in ancestors:
                if a != subj:
                    for f in self.kg.facts:
                        if f.subj == a and f.rel and not f.neg and f.obj:
                            if obj and (obj in f.obj or f.obj in obj):
                                return "yes"
            return "no"
        
        # ── HOW MANY questions ──
        if "how many" in raw:
            count_target = obj if obj else subj
            count_val = self.kg.count("with:" + norm(count_target))
            if count_val > 0:
                return str(count_val)
            count_val = self.kg.count("type:" + norm(count_target))
            if count_val > 0:
                return str(count_val)
            # General entity counting
            if count_target:
                seen = set()
                for f in self.kg.facts:
                    if f.obj == count_target and f.rel == "with" and not f.neg:
                        seen.add(f.subj)
                    if f.subj == count_target and f.rel in LOCATIVE_VERBS and not f.neg:
                        seen.add(f.subj)
                return str(len(seen)) if seen else "0"
            return "0"
        
        # ── Generic fallback ──
        found, neg = self.kg.verify(subj, obj, verb)
        if found and not neg:
            return "yes"
        if neg:
            return "no"
        
        return ""


def normalize_answer(ans: str) -> str:
    """Normalize answer for comparison."""
    return norm(ans).strip().rstrip(".")


def evaluate_task(task_name: str, data_dir: str = "data/babi") -> tuple:
    """Evaluate a single bAbI task."""
    path = f"{data_dir}/{task_name}_test.txt"
    try:
        stories = load_babi_file(path)
    except FileNotFoundError:
        print(f"  ! File not found: {path}")
        return 0, 0
    
    engine = BabiDetailedEngine()
    correct = 0
    total = 0
    
    for story in stories:
        # Reset for each story
        engine.kg.reset()
        
        for entry in story:
            if entry["type"] == "statement":
                engine.hear(entry["text"])
            else:
                total += 1
                resp = engine._answer_babi_question(entry["question"])
                expected = entry["answer"]
                
                if normalize_answer(resp) == normalize_answer(expected):
                    correct += 1
                else:
                    pass  # We'll optionally print failures
    
    return correct, total


def run_full_evaluation():
    """Run full 20-task bAbI evaluation."""
    print("=" * 68)
    print("  bAbI 20 Tasks — Full Evaluation (EntityKnowledgeGraph)")
    print("  Genuine reasoning — no task-specific hardcoding")
    print("=" * 68)
    print()
    
    total_c = total_t = 0
    results = []
    
    for task_id, task_desc in TASKS:
        t0 = time.time()
        c, t = evaluate_task(task_id)
        elapsed = time.time() - t0
        
        pct = 100.0 * c / t if t else 0
        solved = "✓ SOLVED" if pct >= 95.0 else ""
        
        total_c += c
        total_t += t
        results.append((task_id, task_desc, c, t, pct, solved, elapsed))
        
        marker = "✓" if solved else ("*" if pct >= 50 else " ")
        pct_str = f"[{pct:5.1f}%]"
        print(f"  {marker}  {task_id:38s} {c:4d}/{t:<4d}  {pct_str}  {solved:12s}  [{elapsed:4.0f}s]")
        sys.stdout.flush()
    
    print()
    print("=" * 68)
    tot_pct = 100.0 * total_c / total_t if total_t else 0
    solved_n = sum(1 for r in results if r[5])
    print(f"  TOTAL: {total_c}/{total_t} ({tot_pct:.1f}%)  |  Solved: {solved_n}/20")
    print(f"  Total time: {time.time() - results[0][6]:.0f}s" if results else "")
    print("=" * 68)


def run_single_task(task_name: str):
    """Run a single bAbI task with detailed output."""
    path = f"data/babi/{task_name}_test.txt"
    try:
        stories = load_babi_file(path)
    except FileNotFoundError:
        print(f"File not found: {path}")
        return
    
    engine = BabiDetailedEngine()
    correct = 0
    total = 0
    failures = []
    
    for si, story in enumerate(stories):
        engine.kg.reset()
        
        for entry in story:
            if entry["type"] == "statement":
                engine.hear(entry["text"])
            else:
                total += 1
                resp = engine._answer_babi_question(entry["question"])
                expected = entry["answer"]
                
                if normalize_answer(resp) == normalize_answer(expected):
                    correct += 1
                else:
                    failures.append({
                        "story": si,
                        "question": entry["question"],
                        "expected": expected,
                        "got": resp,
                    })
    
    pct = 100.0 * correct / total if total else 0
    print(f"\n{task_name}: {correct}/{total} ({pct:.1f}%) {'✓ SOLVED' if pct >= 95.0 else ''}")
    
    if failures:
        print(f"\nFirst {min(20, len(failures))} failures:")
        for f in failures[:20]:
            print(f"  Q: {f['question']}")
            print(f"  Expected: {f['expected']} | Got: '{f['got']}'")
            print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "all":
            run_full_evaluation()
        else:
            run_single_task(sys.argv[1])
    else:
        run_full_evaluation()
