import spacy
nlp = spacy.load("en_core_web_sm")
sent = "The region is bordered by China to the north and Russia to the northeast, across the Amnok and Duman rivers, and is separated from Japan to the southeast by the Korea Strait."
doc = nlp(sent)
for tok in doc:
    print(f"  {tok.text:15s} {tok.dep_:15s} {tok.lemma_:12s} head={tok.head.text}")
print()
for tok in doc:
    if tok.dep_ in ("ROOT", "conj"):
        print(f"CLAUSE: {tok.text} ({tok.lemma_})")
        for ch in tok.children:
            print(f"  child: {ch.text:15s} dep={ch.dep_:12s}")
        print()
