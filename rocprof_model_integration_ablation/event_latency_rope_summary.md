[
  {
    "variant": "unfused_all",
    "ms": 3.4919012451171874,
    "speedup_vs_unfused": 1.0
  },
  {
    "variant": "k1_fused",
    "ms": 1.6880923461914064,
    "speedup_vs_unfused": 2.0685487100249276
  },
  {
    "variant": "k1_k2_fused",
    "ms": 1.8643278503417968,
    "speedup_vs_unfused": 1.873008143110129
  },
  {
    "variant": "attention_fused",
    "ms": 0.7825678253173828,
    "speedup_vs_unfused": 4.462106838727993
  },
  {
    "variant": "full_fused",
    "ms": 0.031756908893585206,
    "speedup_vs_unfused": 109.95721456449876
  }
]

| Variant | Latency (ms) | Speedup vs unfused |
| --- | ---: | ---: |
| unfused_all | 3.491901 | 1.00x |
| k1_fused | 1.688092 | 2.07x |
| k1_k2_fused | 1.864328 | 1.87x |
| attention_fused | 0.782568 | 4.46x |
| full_fused | 0.031757 | 109.96x |
