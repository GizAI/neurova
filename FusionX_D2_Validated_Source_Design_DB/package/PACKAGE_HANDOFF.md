# Package handoff boundary

The product concept targets chiplets and HBM, but D2 contains no fabricated
interposer or substrate data. The OSAT/package team must supply and sign off:

- die and HBM bump maps, escape routing and microbump/C4 rules;
- interposer/substrate stack-up, extracted package netlist and channel models;
- power-delivery impedance/transient analysis and decoupling placement;
- HBM/PCIe/UCIe SI, crosstalk and timing budgets;
- CFD/thermal interface, cold plate, warpage, stress and reliability models;
- assembly flow, known-good-die/known-good-stack policy, package test and yield model.
