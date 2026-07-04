[
  {
    "variant": "unfused_all",
    "ms": 3.3598590087890625,
    "speedup_vs_unfused": 1.0
  },
  {
    "variant": "k1_fused",
    "ms": 1.6963580322265626,
    "speedup_vs_unfused": 1.9806308249556634
  },
  {
    "variant": "k1_k2_fused",
    "ms": 1.5916622924804686,
    "speedup_vs_unfused": 2.110911984698093
  },
  {
    "variant": "attention_fused",
    "ms": 0.7874147796630859,
    "speedup_vs_unfused": 4.266949383686521
  },
  {
    "variant": "full_fused",
    "ms": 0.03120074987411499,
    "speedup_vs_unfused": 107.68520059117218
  }
]

| Variant | Latency (ms) | Speedup vs unfused |
| --- | ---: | ---: |
| unfused_all | 3.359859 | 1.00x |
| k1_fused | 1.696358 | 1.98x |
| k1_k2_fused | 1.591662 | 2.11x |
| attention_fused | 0.787415 | 4.27x |
| full_fused | 0.031201 | 107.69x |
