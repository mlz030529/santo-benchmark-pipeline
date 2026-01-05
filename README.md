# SANTO Benchmark Pipeline

This repository provides a reproducible benchmarking pipeline for SANTO on large-scale spatial transcriptomics datasets.


---

## SANTO Repository

SANTO is originally developed and maintained by the authors at:

https://github.com/zhanglabtools/SANTO

This makes SANTO easier to benchmark against other spatial alignment methods under the same conditions.

---

## Repository Structure

```text
santo-benchmark-pipeline/
├── run_santo_pairs.py        # Main benchmarking script
├── santo.slurm               # Example SLURM job script
├── environment.yml           # Conda environment
├── README.md
