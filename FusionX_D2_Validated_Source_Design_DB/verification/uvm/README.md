# UVM closure plan

D2 includes directed HDL tests, but no completed UVM coverage claim. Product RTL
freeze requires an environment with:

- transaction-level SQ/CQ, HBM, scratchpad and UCIe agents;
- randomized M/N/K including zero, one, exact-tile and ragged edges;
- all packed data types and scale/out-shift combinations;
- random HBM latency, backpressure, response reordering, ECC poison and retry;
- stream opcode sequences and GEMM-to-stream chaining;
- clock-ratio and asynchronous reset sweeps;
- secure-boot, watchdog, thermal and RAS fault injection;
- scoreboard comparison against the Python/C++ numerical model;
- requirements, functional and code-coverage closure with approved waivers.
