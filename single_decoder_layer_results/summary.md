# Single Decoder Layer Benchmark Summary

## Correctness

| batch | seq_len | ffn_hidden | variant | reference | max_error | mean_error | mean_relative_error | tolerance | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | 11008 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 1 | 10 | 11008 | quarot_unfused | fp16_baseline | 11.9766 | 2.48863 | 249.769 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 1 | 10 | 11008 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 1 | 10 | 11008 | fused_quarot | fp16_baseline | 11.6406 | 2.48861 | 399.48 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 1 | 10 | 11008 | fused_quarot | quarot_unfused | 0.946289 | 0.21026 | 95.8711 | max<=1.25,mean<=0.25 | PASS |
| 1 | 10 | 14336 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 1 | 10 | 14336 | quarot_unfused | fp16_baseline | 12.625 | 2.74503 | 349.477 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 1 | 10 | 14336 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 1 | 10 | 14336 | fused_quarot | fp16_baseline | 12.5156 | 2.74693 | 390.326 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 1 | 10 | 14336 | fused_quarot | quarot_unfused | 1.16797 | 0.232261 | 0.677262 | max<=1.25,mean<=0.25 | PASS |
| 1 | 128 | 11008 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 1 | 128 | 11008 | quarot_unfused | fp16_baseline | 10.8906 | 2.39707 | 1088.98 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 1 | 128 | 11008 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 1 | 128 | 11008 | fused_quarot | fp16_baseline | 10.6367 | 2.39717 | 1044.11 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 1 | 128 | 11008 | fused_quarot | quarot_unfused | 0.429688 | 0.104872 | 0.40764 | max<=1.25,mean<=0.25 | PASS |
| 1 | 128 | 14336 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 1 | 128 | 14336 | quarot_unfused | fp16_baseline | 12.1875 | 2.66196 | 4.4369 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 1 | 128 | 14336 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 1 | 128 | 14336 | fused_quarot | fp16_baseline | 12.2227 | 2.66173 | 4.53485 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 1 | 128 | 14336 | fused_quarot | quarot_unfused | 0.615234 | 0.123141 | 0.51759 | max<=1.25,mean<=0.25 | PASS |
| 1 | 1024 | 11008 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 1 | 1024 | 11008 | quarot_unfused | fp16_baseline | 10.7148 | 2.36058 | 5.31637 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 1 | 1024 | 11008 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 1 | 1024 | 11008 | fused_quarot | fp16_baseline | 10.7422 | 2.36023 | 5.3075 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 1 | 1024 | 11008 | fused_quarot | quarot_unfused | 0.182617 | 0.0311524 | 0.0878045 | max<=1.25,mean<=0.25 | PASS |
| 1 | 1024 | 14336 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 1 | 1024 | 14336 | quarot_unfused | fp16_baseline | 11.8613 | 2.60617 | 14.9313 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 1 | 1024 | 14336 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 1 | 1024 | 14336 | fused_quarot | fp16_baseline | 11.8594 | 2.60614 | 15.0449 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 1 | 1024 | 14336 | fused_quarot | quarot_unfused | 0.179688 | 0.0381183 | 11.8967 | max<=1.25,mean<=0.25 | PASS |
| 1 | 4096 | 11008 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 1 | 4096 | 11008 | quarot_unfused | fp16_baseline | 10.8203 | 2.36623 | 5.14038 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 1 | 4096 | 11008 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 1 | 4096 | 11008 | fused_quarot | fp16_baseline | 10.8281 | 2.36592 | 5.13841 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 1 | 4096 | 11008 | fused_quarot | quarot_unfused | 0.0727539 | 0.0160344 | 9.11248 | max<=1.25,mean<=0.25 | PASS |
| 1 | 4096 | 14336 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 1 | 4096 | 14336 | quarot_unfused | fp16_baseline | 11.7227 | 2.60146 | 4.61374 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 1 | 4096 | 14336 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 1 | 4096 | 14336 | fused_quarot | fp16_baseline | 11.7109 | 2.60131 | 4.60718 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 1 | 4096 | 14336 | fused_quarot | quarot_unfused | 0.119141 | 0.0259561 | 13.2967 | max<=1.25,mean<=0.25 | PASS |
| 2 | 10 | 11008 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 2 | 10 | 11008 | quarot_unfused | fp16_baseline | 11.9766 | 2.35061 | 127.456 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 2 | 10 | 11008 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 2 | 10 | 11008 | fused_quarot | fp16_baseline | 11.9141 | 2.35262 | 202.306 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 2 | 10 | 11008 | fused_quarot | quarot_unfused | 0.946289 | 0.203462 | 48.2021 | max<=1.25,mean<=0.25 | PASS |
| 2 | 10 | 14336 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 2 | 10 | 14336 | quarot_unfused | fp16_baseline | 12.625 | 2.63344 | 178.105 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 2 | 10 | 14336 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 2 | 10 | 14336 | fused_quarot | fp16_baseline | 12.5156 | 2.63841 | 198.522 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 2 | 10 | 14336 | fused_quarot | quarot_unfused | 1.16797 | 0.226249 | 0.597901 | max<=1.25,mean<=0.25 | PASS |
| 2 | 128 | 11008 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 2 | 128 | 11008 | quarot_unfused | fp16_baseline | 12.1172 | 2.28324 | 547.959 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 2 | 128 | 11008 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 2 | 128 | 11008 | fused_quarot | fp16_baseline | 12.1797 | 2.28415 | 525.615 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 2 | 128 | 11008 | fused_quarot | quarot_unfused | 0.554688 | 0.100591 | 0.325635 | max<=1.25,mean<=0.25 | PASS |
| 2 | 128 | 14336 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 2 | 128 | 14336 | quarot_unfused | fp16_baseline | 12.6211 | 2.56033 | 5.0026 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 2 | 128 | 14336 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 2 | 128 | 14336 | fused_quarot | fp16_baseline | 12.6445 | 2.56019 | 5.05771 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 2 | 128 | 14336 | fused_quarot | quarot_unfused | 0.615234 | 0.118061 | 0.575489 | max<=1.25,mean<=0.25 | PASS |
| 2 | 1024 | 11008 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 2 | 1024 | 11008 | quarot_unfused | fp16_baseline | 11.9102 | 2.26755 | 4.95829 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 2 | 1024 | 11008 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 2 | 1024 | 11008 | fused_quarot | fp16_baseline | 11.8789 | 2.26695 | 4.94564 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 2 | 1024 | 11008 | fused_quarot | quarot_unfused | 0.185791 | 0.0334653 | 0.103021 | max<=1.25,mean<=0.25 | PASS |
| 2 | 1024 | 14336 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 2 | 1024 | 14336 | quarot_unfused | fp16_baseline | 12.4258 | 2.52833 | 9.84795 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 2 | 1024 | 14336 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 2 | 1024 | 14336 | fused_quarot | fp16_baseline | 12.4492 | 2.52853 | 9.90175 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 2 | 1024 | 14336 | fused_quarot | quarot_unfused | 0.181641 | 0.0389709 | 6.00134 | max<=1.25,mean<=0.25 | PASS |
| 2 | 4096 | 11008 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 2 | 4096 | 11008 | quarot_unfused | fp16_baseline | 11.875 | 2.27071 | 5.63418 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 2 | 4096 | 11008 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 2 | 4096 | 11008 | fused_quarot | fp16_baseline | 11.8867 | 2.27074 | 5.63135 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 2 | 4096 | 11008 | fused_quarot | quarot_unfused | 0.0727539 | 0.0140206 | 4.57539 | max<=1.25,mean<=0.25 | PASS |
| 2 | 4096 | 14336 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 2 | 4096 | 14336 | quarot_unfused | fp16_baseline | 12.5039 | 2.527 | 5.50546 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 2 | 4096 | 14336 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 2 | 4096 | 14336 | fused_quarot | fp16_baseline | 12.5117 | 2.5268 | 5.49581 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 2 | 4096 | 14336 | fused_quarot | quarot_unfused | 0.119141 | 0.0236872 | 6.68555 | max<=1.25,mean<=0.25 | PASS |
| 4 | 10 | 11008 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 4 | 10 | 11008 | quarot_unfused | fp16_baseline | 13.2617 | 2.44228 | 66.4332 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 4 | 10 | 11008 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 4 | 10 | 11008 | fused_quarot | fp16_baseline | 13.6523 | 2.44255 | 103.895 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 4 | 10 | 11008 | fused_quarot | quarot_unfused | 0.946289 | 0.180926 | 32.7131 | max<=1.25,mean<=0.25 | PASS |
| 4 | 10 | 14336 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 4 | 10 | 14336 | quarot_unfused | fp16_baseline | 13.9414 | 2.72345 | 199.6 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 4 | 10 | 14336 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 4 | 10 | 14336 | fused_quarot | fp16_baseline | 14.2578 | 2.72627 | 196.233 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 4 | 10 | 14336 | fused_quarot | quarot_unfused | 1.16797 | 0.201296 | 4.13826 | max<=1.25,mean<=0.25 | PASS |
| 4 | 128 | 11008 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 4 | 128 | 11008 | quarot_unfused | fp16_baseline | 13.4062 | 2.41574 | 279.691 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 4 | 128 | 11008 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 4 | 128 | 11008 | fused_quarot | fp16_baseline | 13.6172 | 2.41318 | 268.553 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 4 | 128 | 11008 | fused_quarot | quarot_unfused | 0.621582 | 0.111942 | 8.09433 | max<=1.25,mean<=0.25 | PASS |
| 4 | 128 | 14336 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 4 | 128 | 14336 | quarot_unfused | fp16_baseline | 12.8867 | 2.67346 | 5.56888 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 4 | 128 | 14336 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 4 | 128 | 14336 | fused_quarot | fp16_baseline | 12.8125 | 2.67395 | 5.60036 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 4 | 128 | 14336 | fused_quarot | quarot_unfused | 0.754395 | 0.123119 | 15.6061 | max<=1.25,mean<=0.25 | PASS |
| 4 | 1024 | 11008 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 4 | 1024 | 11008 | quarot_unfused | fp16_baseline | 13.4805 | 2.40052 | 5.31648 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 4 | 1024 | 11008 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 4 | 1024 | 11008 | fused_quarot | fp16_baseline | 13.4453 | 2.40023 | 5.20664 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 4 | 1024 | 11008 | fused_quarot | quarot_unfused | 0.210938 | 0.0365091 | 0.120287 | max<=1.25,mean<=0.25 | PASS |
| 4 | 1024 | 14336 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 4 | 1024 | 14336 | quarot_unfused | fp16_baseline | 12.8281 | 2.66403 | 7.98304 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 4 | 1024 | 14336 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 4 | 1024 | 14336 | fused_quarot | fp16_baseline | 12.832 | 2.66342 | 7.99666 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 4 | 1024 | 14336 | fused_quarot | quarot_unfused | 0.181641 | 0.0387372 | 3.04667 | max<=1.25,mean<=0.25 | PASS |
| 4 | 4096 | 11008 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 4 | 4096 | 11008 | quarot_unfused | fp16_baseline | 13.3125 | 2.4013 | 5.74819 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 4 | 4096 | 11008 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 4 | 4096 | 11008 | fused_quarot | fp16_baseline | 13.2695 | 2.4011 | 5.7446 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 4 | 4096 | 11008 | fused_quarot | quarot_unfused | 0.11499 | 0.0188496 | 2.3175 | max<=1.25,mean<=0.25 | PASS |
| 4 | 4096 | 14336 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 4 | 4096 | 14336 | quarot_unfused | fp16_baseline | 12.8594 | 2.66423 | 5.78029 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 4 | 4096 | 14336 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 4 | 4096 | 14336 | fused_quarot | fp16_baseline | 12.8906 | 2.66385 | 5.76444 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 4 | 4096 | 14336 | fused_quarot | quarot_unfused | 0.163086 | 0.023749 | 6.05154 | max<=1.25,mean<=0.25 | PASS |
| 8 | 10 | 11008 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 8 | 10 | 11008 | quarot_unfused | fp16_baseline | 15.1406 | 2.44318 | 101.643 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 8 | 10 | 11008 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 8 | 10 | 11008 | fused_quarot | fp16_baseline | 14.8516 | 2.44344 | 113.581 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 8 | 10 | 11008 | fused_quarot | quarot_unfused | 0.946289 | 0.167554 | 32.5873 | max<=1.25,mean<=0.25 | PASS |
| 8 | 10 | 14336 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 8 | 10 | 14336 | quarot_unfused | fp16_baseline | 13.9922 | 2.7597 | 140.641 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 8 | 10 | 14336 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 8 | 10 | 14336 | fused_quarot | fp16_baseline | 14.2578 | 2.76016 | 150.146 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 8 | 10 | 14336 | fused_quarot | quarot_unfused | 1.16797 | 0.187405 | 3.45263 | max<=1.25,mean<=0.25 | PASS |
| 8 | 128 | 11008 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 8 | 128 | 11008 | quarot_unfused | fp16_baseline | 13.4062 | 2.40533 | 280.841 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 8 | 128 | 11008 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 8 | 128 | 11008 | fused_quarot | fp16_baseline | 13.6172 | 2.40338 | 269.318 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 8 | 128 | 11008 | fused_quarot | quarot_unfused | 0.621582 | 0.0911874 | 4.1876 | max<=1.25,mean<=0.25 | PASS |
| 8 | 128 | 14336 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 8 | 128 | 14336 | quarot_unfused | fp16_baseline | 12.8867 | 2.70991 | 5.75049 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 8 | 128 | 14336 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 8 | 128 | 14336 | fused_quarot | fp16_baseline | 12.8125 | 2.70991 | 5.75996 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 8 | 128 | 14336 | fused_quarot | quarot_unfused | 0.754395 | 0.102372 | 14.9312 | max<=1.25,mean<=0.25 | PASS |
| 8 | 1024 | 11008 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 8 | 1024 | 11008 | quarot_unfused | fp16_baseline | 13.4805 | 2.393 | 101.03 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 8 | 1024 | 11008 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 8 | 1024 | 11008 | fused_quarot | fp16_baseline | 13.4453 | 2.39272 | 101.04 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 8 | 1024 | 11008 | fused_quarot | quarot_unfused | 0.210938 | 0.0310636 | 1.81425 | max<=1.25,mean<=0.25 | PASS |
| 8 | 1024 | 14336 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 8 | 1024 | 14336 | quarot_unfused | fp16_baseline | 12.8359 | 2.70133 | 35.4041 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 8 | 1024 | 14336 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 8 | 1024 | 14336 | fused_quarot | fp16_baseline | 12.8438 | 2.70089 | 34.5882 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 8 | 1024 | 14336 | fused_quarot | quarot_unfused | 0.181641 | 0.0340412 | 2.63758 | max<=1.25,mean<=0.25 | PASS |
| 8 | 4096 | 11008 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 8 | 4096 | 11008 | quarot_unfused | fp16_baseline | 13.3125 | 2.39425 | 162.197 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 8 | 4096 | 11008 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 8 | 4096 | 11008 | fused_quarot | fp16_baseline | 13.2695 | 2.39395 | 161.374 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 8 | 4096 | 11008 | fused_quarot | quarot_unfused | 0.185791 | 0.0220631 | 1.38718 | max<=1.25,mean<=0.25 | PASS |
| 8 | 4096 | 14336 | fp16_baseline | fp16_baseline | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 8 | 4096 | 14336 | quarot_unfused | fp16_baseline | 12.8594 | 2.70166 | 73.279 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 8 | 4096 | 14336 | quarot_unfused | quarot_unfused | 0 | 0 | 0 | max<=1.25,mean<=0.25 | PASS |
| 8 | 4096 | 14336 | fused_quarot | fp16_baseline | 12.8906 | 2.70141 | 72.2194 | max<=1.25,mean<=0.25 | NA_quantization_drift |
| 8 | 4096 | 14336 | fused_quarot | quarot_unfused | 0.210938 | 0.0269354 | 3.07427 | max<=1.25,mean<=0.25 | PASS |

## Latency

| batch | seq_len | ffn_hidden | variant | latency_ms |
| --- | --- | --- | --- | --- |
| 1 | 10 | 11008 | fp16_baseline | 2.23054 |
| 1 | 10 | 11008 | quarot_unfused | 4.57472 |
| 1 | 10 | 11008 | fused_quarot | 0.975986 |
| 1 | 10 | 14336 | fp16_baseline | 1.17378 |
| 1 | 10 | 14336 | quarot_unfused | 4.34819 |
| 1 | 10 | 14336 | fused_quarot | 1.07678 |
| 1 | 128 | 11008 | fp16_baseline | 1.16913 |
| 1 | 128 | 11008 | quarot_unfused | 7.04375 |
| 1 | 128 | 11008 | fused_quarot | 0.976726 |
| 1 | 128 | 14336 | fp16_baseline | 1.1491 |
| 1 | 128 | 14336 | quarot_unfused | 4.24159 |
| 1 | 128 | 14336 | fused_quarot | 1.08232 |
| 1 | 1024 | 11008 | fp16_baseline | 1.42381 |
| 1 | 1024 | 11008 | quarot_unfused | 4.22869 |
| 1 | 1024 | 11008 | fused_quarot | 1.00587 |
| 1 | 1024 | 14336 | fp16_baseline | 1.23498 |
| 1 | 1024 | 14336 | quarot_unfused | 4.64092 |
| 1 | 1024 | 14336 | fused_quarot | 1.12573 |
| 1 | 4096 | 11008 | fp16_baseline | 1.49361 |
| 1 | 4096 | 11008 | quarot_unfused | 5.54217 |
| 1 | 4096 | 11008 | fused_quarot | 1.17662 |
| 1 | 4096 | 14336 | fp16_baseline | 1.52987 |
| 1 | 4096 | 14336 | quarot_unfused | 5.63861 |
| 1 | 4096 | 14336 | fused_quarot | 1.29962 |
| 2 | 10 | 11008 | fp16_baseline | 1.05328 |
| 2 | 10 | 11008 | quarot_unfused | 4.21484 |
| 2 | 10 | 11008 | fused_quarot | 0.969723 |
| 2 | 10 | 14336 | fp16_baseline | 1.15801 |
| 2 | 10 | 14336 | quarot_unfused | 4.23986 |
| 2 | 10 | 14336 | fused_quarot | 1.08293 |
| 2 | 128 | 11008 | fp16_baseline | 1.06347 |
| 2 | 128 | 11008 | quarot_unfused | 4.56435 |
| 2 | 128 | 11008 | fused_quarot | 0.97797 |
| 2 | 128 | 14336 | fp16_baseline | 1.15807 |
| 2 | 128 | 14336 | quarot_unfused | 4.20436 |
| 2 | 128 | 14336 | fused_quarot | 1.07834 |
| 2 | 1024 | 11008 | fp16_baseline | 1.13585 |
| 2 | 1024 | 11008 | quarot_unfused | 4.51411 |
| 2 | 1024 | 11008 | fused_quarot | 1.01963 |
| 2 | 1024 | 14336 | fp16_baseline | 1.27478 |
| 2 | 1024 | 14336 | quarot_unfused | 6.19983 |
| 2 | 1024 | 14336 | fused_quarot | 1.1312 |
| 2 | 4096 | 11008 | fp16_baseline | 1.53872 |
| 2 | 4096 | 11008 | quarot_unfused | 8.63123 |
| 2 | 4096 | 11008 | fused_quarot | 1.19245 |
| 2 | 4096 | 14336 | fp16_baseline | 1.54505 |
| 2 | 4096 | 14336 | quarot_unfused | 8.72011 |
| 2 | 4096 | 14336 | fused_quarot | 1.30813 |
| 4 | 10 | 11008 | fp16_baseline | 1.14141 |
| 4 | 10 | 11008 | quarot_unfused | 5.4374 |
| 4 | 10 | 11008 | fused_quarot | 1.1755 |
| 4 | 10 | 14336 | fp16_baseline | 1.20932 |
| 4 | 10 | 14336 | quarot_unfused | 4.99237 |
| 4 | 10 | 14336 | fused_quarot | 1.0888 |
| 4 | 128 | 11008 | fp16_baseline | 1.14626 |
| 4 | 128 | 11008 | quarot_unfused | 4.33254 |
| 4 | 128 | 11008 | fused_quarot | 0.989345 |
| 4 | 128 | 14336 | fp16_baseline | 1.23308 |
| 4 | 128 | 14336 | quarot_unfused | 4.46997 |
| 4 | 128 | 14336 | fused_quarot | 1.09481 |
| 4 | 1024 | 11008 | fp16_baseline | 1.15471 |
| 4 | 1024 | 11008 | quarot_unfused | 5.24927 |
| 4 | 1024 | 11008 | fused_quarot | 1.03607 |
| 4 | 1024 | 14336 | fp16_baseline | 1.2684 |
| 4 | 1024 | 14336 | quarot_unfused | 5.37472 |
| 4 | 1024 | 14336 | fused_quarot | 1.13552 |
| 4 | 4096 | 11008 | fp16_baseline | 1.48977 |
| 4 | 4096 | 11008 | quarot_unfused | 14.7834 |
| 4 | 4096 | 11008 | fused_quarot | 1.20974 |
| 4 | 4096 | 14336 | fp16_baseline | 1.59991 |
| 4 | 4096 | 14336 | quarot_unfused | 14.8379 |
| 4 | 4096 | 14336 | fused_quarot | 1.31823 |
| 8 | 10 | 11008 | fp16_baseline | 1.08384 |
| 8 | 10 | 11008 | quarot_unfused | 9.26758 |
| 8 | 10 | 11008 | fused_quarot | 1.46965 |
| 8 | 10 | 14336 | fp16_baseline | 1.17847 |
| 8 | 10 | 14336 | quarot_unfused | 4.22893 |
| 8 | 10 | 14336 | fused_quarot | 1.10971 |
| 8 | 128 | 11008 | fp16_baseline | 1.09862 |
| 8 | 128 | 11008 | quarot_unfused | 4.21053 |
| 8 | 128 | 11008 | fused_quarot | 1.00636 |
| 8 | 128 | 14336 | fp16_baseline | 1.20962 |
| 8 | 128 | 14336 | quarot_unfused | 4.24423 |
| 8 | 128 | 14336 | fused_quarot | 1.11543 |
| 8 | 1024 | 11008 | fp16_baseline | 1.40049 |
| 8 | 1024 | 11008 | quarot_unfused | 8.57482 |
| 8 | 1024 | 11008 | fused_quarot | 1.05363 |
| 8 | 1024 | 14336 | fp16_baseline | 1.46491 |
| 8 | 1024 | 14336 | quarot_unfused | 8.63737 |
| 8 | 1024 | 14336 | fused_quarot | 1.17057 |
| 8 | 4096 | 11008 | fp16_baseline | 2.19815 |
| 8 | 4096 | 11008 | quarot_unfused | 27.4995 |
| 8 | 4096 | 11008 | fused_quarot | 1.29973 |
| 8 | 4096 | 14336 | fp16_baseline | 2.30341 |
| 8 | 4096 | 14336 | quarot_unfused | 27.6053 |
| 8 | 4096 | 14336 | fused_quarot | 1.39633 |
