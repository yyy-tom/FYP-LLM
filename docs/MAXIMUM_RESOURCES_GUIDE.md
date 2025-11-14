# Maximum Resources Configuration Guide

Based on your cluster configuration, here are the options for using maximum resources:

## Option 1: Maximum Time (168 hours) - `batch_168h` ⭐

**Best for**: Very long training runs, experiments that need maximum time

```bash
sbatch scripts/train_job_max.sh
```

**Resources:**
- **Partition**: `batch_168h`
- **QoS**: `ex_batch`
- **Time Limit**: 168 hours (7 days)
- **GPUs**: 8 (maximum)
- **CPUs**: 80 (recommended 10:1 ratio for most GPUs)
- **Priority**: Very High
- **Interactive**: No (batch only)
- **Max Jobs**: 1 per user

**Note**: This queue has very high priority but only allows 1 job per user.

## Option 2: Maximum Time with Interactive (72 hours) - `gpu_72h`

**Best for**: Long training runs where you might want to monitor/interact

```bash
sbatch scripts/train_job_max_gpu_72h.sh
```

**Resources:**
- **Partition**: `gpu_72h`
- **QoS**: `gpu`
- **Time Limit**: 72 hours (3 days)
- **GPUs**: 8 (maximum)
- **CPUs**: 80 (recommended 10:1 ratio)
- **Priority**: Low
- **Interactive**: Yes
- **Max Jobs**: 4 per user

## Resource Comparison

| Partition | Time Limit | GPUs | CPUs | Priority | Interactive | Max Jobs/User |
|-----------|------------|------|------|----------|-------------|---------------|
| `batch_168h` | 168h (7d) | 8 | 80 | Very High | No | 1 |
| `gpu_72h` | 72h (3d) | 8 | 80 | Low | Yes | 4 |
| `gpu_24h` | 24h (1d) | 8 | 80 | Low | Yes | 4 |
| `gpu_8h` | 8h | 8 | 80 | Normal | Yes | 4 |

## CPU:GPU Ratio Recommendations

Based on your GPU types:
- **Titan X (gpu7-9)**: 6:1 ratio → 48 CPUs max
- **Titan V/XP, RTX 2080 (gpu24-53)**: 10:1 ratio → 80 CPUs max
- **Titan RTX (gpu54-59)**: No limit (recommend 20:1) → 160 CPUs max

The scripts use **80 CPUs** which works for most GPU types. If you get Titan RTX nodes, you can increase to 160 CPUs.

## GPU Selection

To request specific GPU types (if available):

```bash
# Request RTX 2080 (better than Titan X)
sbatch --constraint="rtx2080" scripts/train_job_max.sh

# Request Titan RTX (best available)
sbatch --constraint="titanrtx" scripts/train_job_max.sh
```

## Usage Examples

### Submit Maximum Resources Job

```bash
# Option 1: Maximum time (168 hours)
sbatch scripts/train_job_max.sh

# Option 2: 72 hours with interactive capability
sbatch scripts/train_job_max_gpu_72h.sh
```

### Monitor Job

```bash
# Check job status
squeue -u $USER

# View output in real-time
tail -f logs/train_max_JOBID.out

# Use helper script
./scripts/monitor_job.sh JOBID
```

### Check Resource Usage

```bash
# After job starts, SSH to the node and check
ssh $SLURM_NODELIST
nvidia-smi  # Check GPU usage
htop        # Check CPU usage
```

## Recommendations

1. **For Production Training**: Use `batch_168h` (Option 1) - maximum time, highest priority
2. **For Development/Debugging**: Use `gpu_72h` (Option 2) - can monitor, multiple jobs allowed
3. **For Quick Tests**: Use `gpu_8h` or `gpu_24h` - faster queue time

## Notes

- **batch_168h** has very high priority but only 1 job per user
- **gpu_72h** allows 4 jobs per user but lower priority
- All scripts request 8 GPUs (your maximum)
- CPU count (80) is optimized for 10:1 ratio (works for most GPUs)
- All cache directories are set to your quota path automatically

