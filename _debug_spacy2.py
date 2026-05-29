import spacy
nlp = spacy.load("en_core_web_sm")

# Division sentence
doc = nlp("it has been politically divided at or near the 38th parallel between North Korea and South Korea")
print("=== DIVISION ===")
for tok in doc:
    if tok.dep_ in ("ROOT","conj") or (tok.pos_ == "VERB"):
        print(f"  {tok.text:15s} dep={tok.dep_:15s} lemma={tok.lemma_:15s}")
        for ch in tok.children:
            print(f"    child: {ch.text:15s} dep={ch.dep_:15s}")
print()

# Border sentence  
doc2 = nlp("The region is bordered by China to the north and Russia to the northeast")
print("=== BORDER ===")
for tok in doc2:
    if tok.dep_ in ("ROOT","conj") or (tok.pos_ == "VERB" and tok.dep_ != "auxpass"):
        print(f"  {tok.text:15s} dep={tok.dep_:15s} lemma={tok.lemma_:15s}")
        for ch in tok.children:
            print(f"    child: {ch.text:15s} dep={ch.dep_:15s}")
            for gc in ch.children:
                print(f"      grand: {gc.text:15s} dep={gc.dep_:15s}")
print()

# "consisting of" sentence
doc3 = nlp("Korea is a peninsular region in East Asia consisting of the Korean Peninsula, Jeju Island, and smaller islands")
print("=== CONSISTING ===")
for tok in doc3:
    if tok.dep_ in ("ROOT","xcomp","advcl","conj") or (tok.pos_ == "VERB" and tok.dep_ != "aux"):
        print(f"  {tok.text:15s} dep={tok.dep_:15s} lemma={tok.lemma_:15s}")
        for ch in tok.children:
            print(f"    child: {ch.text:15s} dep={ch.dep_:15s}")
            for gc in ch.children:
                print(f"      grand: {gc.text:15s} dep={gc.dep_:15s}")
