# MLRift

A systems language for machine-learning workloads — built on
[KernRift](https://github.com/Pantelis23/KernRift) at commit `6cf758b`
(v2.8.15). The compiler's own source is KernRift; MLRift extends the
KRIR backend with ML-specific primitives (tensors, event streams,
continuous-time dynamics, sparse CSR ops, plasticity rules) and will
introduce a `.mlr` frontend for user programs as the roadmap lands.

**Status:** day zero. The rename pass is done (product identity is
MLRift, binary is `mlrc`, bootstrap binary is `build/mlrc`), 436/436
tests pass, self-host fixed point holds. The MLRift-specific syntax
and IR extensions are not started yet — those follow the roadmap in
`~/Desktop/Projects/Work/ideas/MLRift.md`.

## Build

```
make build    # self-compiles build/mlrc (bootstrap committed)
make test     # 436/436
make bootstrap   # verify stage3 == stage4
mlrc --version
```

## Why "built on KernRift"

MLRift is explicitly a **layer on top of KernRift**, not a hard fork.
It shares the type system, the optimization pipeline, the codegen
backends (x86_64 + ARM64, Linux/macOS/Windows/Android), and all the
infrastructure KernRift spent the last year hardening. MLRift-specific
work lives in added passes, added IR ops, and a new frontend — not in
re-implementing the basics. When KernRift fixes a backend bug,
MLRift inherits it with a cherry-pick.

## License

Same as KernRift — see `LICENSE`.
