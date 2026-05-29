import re

with open("neurova/reasoner.py", "r") as f:
    content = f.read()

new_logic = """
        if not steps:
            broad = self.memory.retrieve_broad(subject)
            if broad:
                steps = [ProofStepIR(conclusion=b, rule_applied="contextual_association", verifier_status="associated") for b in broad]
                return ProofIR(query=f"what is {subject}", success=True, status="proved_by_association", steps=steps, active_memory_trace=["retrieve_broad", f"zoom_in=claims_about:0", f"broad_context:{len(broad)}", "verifier=context"])
                
        return ProofIR"""

content = content.replace("        return ProofIR(query=f\"what is {subject}\"", new_logic.lstrip('\n'))

with open("neurova/reasoner.py", "w") as f:
    f.write(content)
