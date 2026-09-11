# D2 architecture

The active hierarchy is:

```
Host CSR + SQ/CQ
  -> queue manager
  -> host/core async command FIFO
  -> opcode router
       -> tensor engine
          -> packed A/B loaders
          -> Q8.8 outer-product MXU
          -> accumulator requantization
       -> stream engine
          -> RMSNorm / Softmax / RoPE / vector
          -> recurrent state / Top-K / MTP / gather
          -> Patchify3D / AdaLN
  -> 32-bank local scratchpad or external device memory
  -> core/host async completion FIFO
  -> CQ writer + interrupt
```

The D2 RTL intentionally implements one parameterized reference engine in the
integrated top. Datacenter replication, chiplet count, and final physical MAC
lane count are configuration and PPA decisions, not inferred by multiplying
uninstantiated architecture parameters.
