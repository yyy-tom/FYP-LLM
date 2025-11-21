# Quick Start: Working SRUN Commands

## ✅ WORKING Commands (Tested)

### Interactive Access with 10 CPUs (for testing/setup)

**Use nodes in standard `gpu_24h` partition with `highcpucount` feature:**

```bash
# RTX 2080 nodes (gpu40-51) - RECOMMENDED
srun -p gpu_24h --qos gpu --account gpu --gres=gpu:1 --cpus-per-task=10 --constraint=highcpucount --nodelist=gpu40 --pty /bin/bash

# Alternative nodes (if gpu40 is busy)
srun -p gpu_24h --qos gpu --account gpu --gres=gpu:1 --cpus-per-task=10 --constraint=highcpucount --nodelist=gpu41 --pty /bin/bash
srun -p gpu_24h --qos gpu --account gpu --gres=gpu:1 --cpus-per-task=10 --constraint=highcpucount --nodelist=gpu42 --pty /bin/bash
```

### For 80 CPUs (Batch Job Required)

```bash
# Submit batch job
sbatch scripts/train_job_gpu.sh
```

## ❌ Common Errors and Solutions

### Error: "Requested nodes not in this partition"
**Solution**: Use nodes that are actually in the partition you specify.

- `gpu54` is in `ct2401` partition (custom), not `gpu_24h`
- Use `gpu40-51` for `gpu_24h` partition
- Or use `gpu54` with `ct2401` partition (if you have access)

### Error: "CPU:GPU Ratio > 10:1"
**Solution**: Use batch job (sbatch) with `ex_batch` partition and `ex_gpu` QOS.

### Error: "CPU:GPU Ratio > 6:1 and <= 10:1"
**Solution**: Add `--constraint=highcpucount` flag.

## Node Availability by Partition

### gpu_24h partition (idle nodes):
- `gpu7-9`, `gpu24`, `gpu33-37`, `gpu40-51`, `gpu57`
- `projgpu3-6`, `projgpu27-28`

### gpu_8h partition (idle nodes):
- `gpu7-9`, `gpu33-37`, `gpu40-51`, `gpu57`

### Nodes with highcpucount feature:
- `gpu24-29` (Titan V)
- `gpu30-35` (Titan XP)
- `gpu36-39` (RTX 2080)
- `gpu40-53` (RTX 2080) ✅ **Use these!**
- `gpu54-59` (Titan RTX) - but in custom partitions

## Recommended: Use gpu40-51

These nodes are:
- ✅ In standard `gpu_24h` partition
- ✅ Have `highcpucount` feature
- ✅ RTX 2080 (8-11GB VRAM)
- ✅ Currently idle
- ✅ Work with standard srun commands

## Quick Test Command

```bash
# This should work:
srun -p gpu_24h --qos gpu --account gpu --gres=gpu:1 --cpus-per-task=10 --constraint=highcpucount --nodelist=gpu40 --pty /bin/bash

# Once inside, verify:
nvidia-smi
```

