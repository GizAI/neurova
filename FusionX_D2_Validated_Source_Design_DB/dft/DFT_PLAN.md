# DFT production requirements

No DFT insertion or ATPG coverage is claimed in D2. Foundry release requires:

- scan insertion for every clock/power domain, lockups and chain balancing;
- scan compression, test-point insertion and at-speed clock control;
- SRAM MBIST and BISR linked to the actual compiler macros and repair fuses;
- IEEE 1149.1 boundary scan and an IJTAG network for embedded instruments;
- stuck-at, transition and cell-aware ATPG with foundry-agreed targets;
- PLL/SerDes/HBM/PCIe/UCIe test modes and package interconnect tests;
- ATE vectors, fault dictionaries, diagnosis database, wafer-sort and final-test bins.
