# Release gates

A release may be called **front-end RTL frozen** only after real HDL
compile/elaboration, lint, CDC/RDC, formal, UVM/coverage, synthesis and formal
equivalence all pass.

A release may be called **foundry-ready** only after the proprietary handoff
requirements are present and P&R/STA/EMIR/SI/DRC/LVS/DFT/package sign-off is
closed. A final GDS/OASIS file without those inputs is not accepted as evidence.
