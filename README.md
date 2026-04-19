# MLRift

A systems language for machine-learning workloads — forked from
[KernRift](https://github.com/Pantelis23/KernRift) at commit `6cf758b`.

**Status:** day zero. The compiler is still KernRift under the hood
(same source, same `krc` toolchain names, same `.kr` file extension).
The fork exists so MLRift can diverge toward tensors, SIMD, and
ML-specific primitives without destabilising KernRift, which is
staying systems-focused.

## Build

```
make build    # bootstraps from build/krc2 into build/krc2.new
make test     # 436/436 on the KernRift base
```

## License

Same as KernRift — see `LICENSE`.
