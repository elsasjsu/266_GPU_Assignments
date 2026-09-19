# HW2.5 GPU Benchmarking - NVIDIA GeForce RTX 4090

This repository contains the code, measurements, plots, and documentation for HW2.5 GPU benchmarking.

## GPU and software environment

- GPU: NVIDIA GeForce RTX 4090
- GPU UUID: `GPU-3094e1cd-3205-910b-9555-93d8401b1ce0`
- Driver: 610.60
- VRAM: 24564 MiB
- Power limit: 450 W
- PyTorch: 2.1.2
- CUDA used by PyTorch: 12.1
- cuDNN: 8902

## Experiments

The assignment includes:

1. GPU hardware/software provenance using `nvidia-smi -q`.
2. Dense matrix-multiplication benchmarks for FP32, TF32, FP16, and BF16.
3. Memory-bandwidth and arithmetic-intensity measurements.
4. Naive and fused scaled dot-product attention benchmarks.
5. Attention-memory scaling and OOM-boundary investigation.
6. A 20-minute sustained-load experiment with 5-second GPU telemetry sampling.
7. Final summary metrics and plots.

## Main results

- Peak measured BF16 throughput: 180.73 TFLOPS
- BF16 theoretical-reference comparison: 109.4% using the selected 165.2 TFLOPS dense BF16 reference; see `METRICS.md` for the interpretation caveat.
- Effective bandwidth: 912.72 GB/s
- Naive attention controlled OOM boundary: 26,752 success / 26,880 OOM using a 90% PyTorch per-process memory cap.
- Sustained throughput / initial peak: 93.30%
- No clear thermal-throttling onset was observed; the GPU reached the 450 W power ceiling at about 5.08 s.

## Files

- `GPU Assignment 1.ipynb` - main assignment notebook
- `OOM_Test.ipynb` - additional controlled OOM-boundary test
- `results/` - CSV measurements and GPU provenance
- `figures/` - benchmark and thermal plots
- `METRICS.md` - final metrics and discussion
- `RUN_LOG.txt` - experiment log
- `AI_USE.md` - disclosure of AI assistance
- `reservation/GPU_HOURS.md` - GPU reservation and usage record

## Reproducibility note

All reported measurements are from the UUID-labelled RTX 4090 above. The OOM refinement used a 90% PyTorch per-process memory cap because the Windows/Docker environment allowed memory oversubscription beyond the card's nominal physical VRAM.
