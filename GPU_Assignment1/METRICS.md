# HW2.5 Metrics

## GPU provenance

- GPU: NVIDIA GeForce RTX 4090
- GPU UUID: `GPU-3094e1cd-3205-910b-9555-93d8401b1ce0`
- Driver: 610.60
- VRAM: 24564 MiB
- Power limit: 450 W
- PyTorch: 2.1.2
- CUDA: 12.1
- cuDNN: 8902

## Final summary

| Measurement | Result | Notes |
|---|---:|---|
| Peak achieved TFLOPS (BF16) | **180.73 TFLOPS** | Highest UUID-labelled BF16 run |
| % of theoretical peak (BF16) | **109.4%*** | Uses a 165.2 TFLOPS dense BF16 reference; execution/accumulation convention may differ |
| Effective bandwidth | **912.72 GB/s** | FP32 elementwise-add benchmark |
| Naive attention OOM boundary | **26,752 success / 26,880 OOM** | Refined using a 90% PyTorch per-process memory cap |
| Fused attention OOM boundary | **Not yet measured in the current saved summary** | Update this row if a fused OOM refinement was completed separately |
| Steady-state / peak throughput | **93.30%** | 157.82 TFLOPS final-5-min mean / 169.15 TFLOPS first-30-s peak |
| Throttle onset | **None observed** | Near-power-limit operation began at about 5.08 s |

\* The BF16 percentage uses the selected 165.2 TFLOPS dense BF16 reference. Because the measured value exceeds that reference, the result should be treated as a convention mismatch rather than literal efficiency above the hardware maximum.

## Matrix multiplication

The reduced-precision FP16 and BF16 benchmarks achieved the highest throughput because Tensor Cores can accelerate these data types. Larger matrices approached a throughput plateau, whereas smaller matrices underutilized the GPU because launch overhead, scheduling overhead, and insufficient parallel work represented a larger fraction of execution time.

## Memory bandwidth and arithmetic intensity

The FP32 elementwise-add benchmark achieved **912.72 GB/s** with an arithmetic intensity of approximately **0.0833 FLOP/byte**, so it behaves as a memory-bound workload.

Dense matrix multiplication has much higher arithmetic intensity and is primarily compute-bound for sufficiently large matrix sizes.

## Attention

The naive attention implementation explicitly materializes the full `L x L` attention matrix, producing approximately quadratic memory growth with sequence length. The fitted quadratic coefficient from the measured data was approximately `2.98e-8 GiB/token^2`.

At sequence length 16,384:

- Naive peak memory: approximately **8.09 GiB**
- Fused peak memory: approximately **0.086 GiB**
- Naive latency: approximately **28.76 ms**
- Fused latency: approximately **11.84 ms**
- Fused speedup: approximately **2.43x**

The fused implementation avoids fully materializing the attention-score matrix in global memory, reducing intermediate storage and memory traffic.

### Controlled naive-attention OOM boundary

The Windows/Docker environment permitted memory oversubscription beyond nominal VRAM, so a 90% PyTorch per-process memory cap was used for a reproducible boundary test.

- Largest successful sequence length: **26,752**
- Peak allocation at that length: approximately **21.44 GiB**
- Smallest tested failing sequence length: **26,880**

Therefore, under this controlled memory cap, the refined naive-attention OOM boundary lies between **26,752 and 26,880 tokens**.

## Sustained-load experiment

The GPU was stressed for 20 minutes while telemetry was sampled every 5 seconds.

- Peak throughput during first 30 s: **169.15 TFLOPS**
- Mean throughput during final 5 min: **157.82 TFLOPS**
- Steady-state / peak: **93.30%**
- Maximum temperature: **76 C**
- Maximum measured power: **450.17 W**
- First high-utilization sample near the 450 W power limit: approximately **5.08 s**
- Mean clock, first 30 s: **2175 MHz**
- Mean clock, final 5 min: **2429.5 MHz**
- Mean temperature, first 30 s: **56.17 C**
- Mean temperature, final 5 min: **74.95 C**
- Mean power, first 30 s: **376.74 W**
- Mean power, final 5 min: **449.46 W**

No clear sustained thermal-throttling event was observed. The GPU remained at a high graphics clock while temperature increased, and steady-state power was close to the configured 450 W limit. The behavior is therefore more consistent with operation near the power ceiling than with a thermal-temperature ceiling.
