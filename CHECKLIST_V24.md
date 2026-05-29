# BrainOS V24 Checklist — Cognitive Construction Grammar

Scope: resolve the remaining issues from the v22/v23 critique by replacing checklist-specific regex patches with a more principled construction grammar/generalization layer.

## Research-grounded design targets

- [x] Use construction grammar as form-meaning pairings, not endless regex patches.
- [x] Add feature-structure style construction objects with slots, constraints, variants, examples, counterexamples, confidence, and support count.
- [x] Implement unification-style slot binding before IR instantiation.
- [x] Generate construction variants: declarative, question, modal question, negation, and selected inverse/reverse variants.
- [x] Keep no-LLM/no-next-token/no-generation principle.
- [x] Integrate construction grammar before generic regex/template parsers.
- [x] Keep legacy construction learner only as compatibility fallback.

## Critique-driven unresolved issues

- [x] Natural correction: `When I say "A dominates B", it means A is greater than B.`
- [x] Natural correction: `"A is slightly ahead of B" means A greater_than B`.
- [x] Learned construction generalizes to assertion: `alice dominates bob`.
- [x] Learned construction generalizes to question: `is alice greater than bob?`.
- [x] Learned construction generalizes by optional modifiers: `A is slightly ahead of B` -> `A is ahead of B`.
- [x] Learned construction question variant: `is A ahead of B?`.
- [x] Taxonomy question paraphrase: `Is it fair to call Kibo a machine?`.
- [x] Taxonomy question paraphrase: `Could Kibo be treated as a kind of machine?`.
- [x] Taxonomy question paraphrase: `Would Kibo qualify as a machine?`.
- [x] Korean particle variant: `철수는 영희에 비해 우위에 있다`.
- [x] Korean question variant: `철수가 영희보다 앞서 있니?`.
- [x] Korean negated comparison: `철수는 영희보다 크지 않다`.
- [x] Temporal interval: `Alice was CEO from 2025 through 2026.`
- [x] Temporal negation with punctuation: `In 2026, Alice was not the CEO.`
- [x] Temporal overlap contradiction returns inconsistent evidence.
- [x] Exception discourse: `Penguins are birds; however, they usually do not fly.`
- [x] Exception question: `Can a penguin fly even though it is a bird?` returns blocked by exception.
- [x] Event frame variant: `Alice gave a book to Bob in Seoul yesterday.`
- [x] Event frame variant: `Bob received a book from Alice yesterday.`
- [x] Event-derived possession query: `does bob have book?`.

## Verification

- [x] AST parse passes.
- [x] pytest passes.
- [x] run_smoke passes 72/72.
- [x] ZIP re-extract verification passes.
