# gfx1201 Counter Probe

Date: 2026-07-04

Purpose: verify whether `rocprofv3` can provide non-zero memory, LDS, and occupancy counters on the current `gfx1201` GPU.

Workload: `python3 profiler_workloads.py k2_i4 --iters 5`

Kernel filter: `BatchDecodeWithPagedKVCacheKernel`

## rocprof-compute

`rocprof-compute` dependencies were installed into:

```bash
/workspace/.rocprof_compute_deps
```

With `PYTHONPATH=/workspace/.rocprof_compute_deps`, `rocprof-compute --help` runs successfully. However, `rocprof-compute -s` reports:

```text
ERROR Cannot find a supported arch in rocminfo
```

The supported arch list shown by this ROCm 7.2 container is:

```text
gfx908, gfx90a, gfx940, gfx941, gfx942, gfx950
```

Current GPU is `gfx1201`, so official `rocprof-compute` roofline/memory analysis is not available for this machine with the current toolchain.

## Split rocprofv3 Counter Results

The original all-in-one counter set exceeded hardware collection capability. The probes were split into smaller passes.

### Occupancy / Activity

File: `occ_counter_collection.csv`

```text
CU_NUM: 320.0
SIMD_NUM: 640.0
GRBM_COUNT: 3455448.0
GRBM_GUI_ACTIVE: 3455448.0
SQ_WAVES_sum: 160.0
SQ_WAVE_CYCLES: 0.0
```

Usable: `SQ_WAVES_sum`, `GRBM_*`, `CU_NUM`, `SIMD_NUM`

Not usable for runtime occupancy: `SQ_WAVE_CYCLES` is zero, so `MeanOccupancyPerCU` / `OccupancyPercent` cannot be reconstructed.

### Global Memory

Files: `mem_fetch_counter_collection.csv`, `mem_read_counter_collection.csv`, `mem_write_counter_collection.csv`

```text
FetchSize: 0.0
GL2C_EA_RDREQ_32B_sum: 0.0
GL2C_EA_RDREQ_64B_sum: 0.0
GL2C_EA_RDREQ_128B_sum: 0.0
GL2C_EA_WRREQ_64B_sum: 0.0
```

Not usable: GL2C memory counters report zero even when collected in separate passes.

### LDS

File: `lds_raw_counter_collection.csv`

```text
SQ_INSTS_LDS: 0.0
SQC_LDS_IDX_ACTIVE: 0.0
SQC_LDS_BANK_CONFLICT: 0.0
```

Not usable: LDS counters report zero.

## Conclusion

The profiling environment is now set up, but `gfx1201` is not supported by `rocprof-compute` in this ROCm 7.2 container, and `rocprofv3` does not provide non-zero GL2C/LDS/SQ_WAVE_CYCLES counters for this workload/GPU combination.

For the report, use:

- measured kernel time and kernel-call count from `rocprofv3`
- measured `SQ_WAVES_sum` and GRBM activity as activity proxies
- analytical `Min semantic IO` for memory traffic lower bounds

