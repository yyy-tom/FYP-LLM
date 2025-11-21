# Running Training on Remote GPC Cluster

This guide explains how to run training jobs that continue even after disconnecting from the remote terminal.

## Option 1: SLURM Batch Job (Recommended) ⭐

**Best for**: Long training runs, automatic resource management, job queuing

### Step 1: Create the batch script

The `train_job.sh` script is already created for you.

### Step 2: Make it executable

```bash
chmod +x train_job.sh
```

### Step 3: Submit the job

```bash
sbatch train_job.sh
```

### Step 4: Monitor the job

**Real-time monitoring (like watching terminal output):**

```bash
# Method 1: Use the helper script (easiest)
./scripts/monitor_job.sh JOBID

# Method 2: Manual tail (follow output in real-time)
tail -f logs/train_JOBID.out

# Method 3: Watch both output and errors
tail -f logs/train_JOBID.out logs/train_JOBID.err

# Method 4: Check job status
squeue -u $USER

# Cancel job if needed
scancel JOBID
```

**Note**: Unlike tmux, you cannot "switch into" a batch job - it's non-interactive. But you CAN watch the output in real-time using `tail -f`!

### Advantages:

- ✅ Job continues even if you disconnect
- ✅ Automatic resource allocation
- ✅ Logs saved to files automatically
- ✅ Can queue multiple jobs
- ✅ Best practice for HPC clusters

---

## Option 2: tmux (Terminal Multiplexer)

**Best for**: Interactive sessions where you want to reconnect later

### Step 1: Start tmux session

```bash
tmux new -s qwen_training
```

### Step 2: Run your training

```bash
python train_qwen_counsel.py --config config.json
```

### Step 3: Detach from session

Press `Ctrl+B`, then `D` (detach)

### Step 4: Reconnect later

```bash
tmux attach -t qwen_training
```

### Or use the helper script:

```bash
chmod +x run_training_tmux.sh
./run_training_tmux.sh
```

### Useful tmux commands:

```bash
# List all sessions
tmux ls

# Attach to session
tmux attach -t qwen_training

# Kill session
tmux kill-session -t qwen_training

# Create new session
tmux new -s session_name
```

---

## Option 3: screen (Terminal Multiplexer)

**Best for**: Similar to tmux, alternative terminal multiplexer

### Step 1: Start screen session

```bash
screen -S qwen_training
```

### Step 2: Run your training

```bash
python train_qwen_counsel.py --config config.json
```

### Step 3: Detach from session

Press `Ctrl+A`, then `D` (detach)

### Step 4: Reconnect later

```bash
screen -r qwen_training
```

### Or use the helper script:

```bash
chmod +x run_training_screen.sh
./run_training_screen.sh
```

### Useful screen commands:

```bash
# List all sessions
screen -ls

# Attach to session
screen -r qwen_training

# Kill session
screen -S qwen_training -X quit
```

---

## Option 4: nohup (Simple Background Process)

**Best for**: Quick and simple, but less control

### Run with nohup:

```bash
nohup python train_qwen_counsel.py --config config.json > training.log 2>&1 &
```

### Check if running:

```bash
ps aux | grep train_qwen_counsel
```

### View output:

```bash
tail -f training.log
```

### Kill process:

```bash
# Find process ID
ps aux | grep train_qwen_counsel

# Kill it
kill PID
```

---

## Option 5: Interactive srun with tmux/screen (Best of Both Worlds!) ⭐

**Best for**: When you want interactive control (like tmux) but with SLURM resource management

This gives you the ability to "switch into" your job like tmux, but with proper SLURM allocation.

### Quick Start (Recommended):

```bash
# Use the helper script
chmod +x scripts/run_interactive_training.sh
./scripts/run_interactive_training.sh
```

### Manual Method:

#### Step 1: Request interactive session

```bash
srun -p gpu_8h --qos gpu --account gpu --gres=gpu:8 --cpus-per-task=8 --pty /bin/bash
```

#### Step 2: Inside the session, start tmux/screen

```bash
tmux new -s training
# or
screen -S training
```

#### Step 3: Run training inside tmux/screen

```bash
python scripts/train_qwen_counsel.py --config configs/config.json
```

#### Step 4: Detach and disconnect

- Detach from tmux/screen: Press `Ctrl+B`, then `D` (for tmux) or `Ctrl+A`, then `D` (for screen)
- You can now disconnect - the job continues running!

#### Step 5: Reconnect later

```bash
# First, get back into the SLURM session (if still running)
# Then attach to tmux
tmux attach -t training
```

### Advantages:

- ✅ Interactive - you can "switch into" the job like tmux
- ✅ SLURM resource management - proper GPU allocation
- ✅ Can detach/reconnect anytime
- ✅ See output in real-time
- ✅ Can interact with the process if needed

---

## Comparison

| Method          | Persistence  | Interactive               | Real-time Output   | Resource Management | Best For                  |
| --------------- | ------------ | ------------------------- | ------------------ | ------------------- | ------------------------- |
| **sbatch**      | ✅ Excellent | ❌ No (can't "switch in") | ✅ Yes (`tail -f`) | ✅ Automatic        | Production runs           |
| **srun + tmux** | ✅ Excellent | ✅ Yes (can "switch in")  | ✅ Yes (live)      | ✅ Automatic        | Interactive debugging     |
| **tmux only**   | ✅ Excellent | ✅ Yes                    | ✅ Yes (live)      | ❌ Manual           | Local/interactive         |
| **screen**      | ✅ Excellent | ✅ Yes                    | ✅ Yes (live)      | ❌ Manual           | Local/interactive         |
| **nohup**       | ✅ Good      | ❌ No                     | ✅ Yes (`tail -f`) | ❌ Manual           | Quick tests               |
| **srun only**   | ❌ No        | ✅ Yes                    | ✅ Yes (live)      | ✅ Automatic        | Quick tests (disconnects) |

### Key Differences:

**sbatch (Batch Job):**

- ✅ Can watch output in real-time with `tail -f logs/train_JOBID.out`
- ❌ Cannot "switch into" the job (non-interactive)
- ✅ Best for production - runs automatically, logs everything
- ✅ Job continues even if you disconnect

**srun + tmux (Interactive with SLURM):**

- ✅ Can watch output in real-time (live in terminal)
- ✅ CAN "switch into" the job (interactive)
- ✅ Best for debugging - you can interact, interrupt, modify
- ✅ Job continues even if you disconnect (detach from tmux first)

---

## Recommended Workflow

1. **For production training**: Use `sbatch train_job.sh` (Option 1)
2. **For debugging/interactive**: Use `srun` + `tmux` (Option 5)
3. **For quick tests**: Use `nohup` (Option 4)

---

## Troubleshooting

### Job gets killed when disconnecting

- Make sure you're using one of the methods above (sbatch, tmux, screen, or nohup)
- Don't just run `python train_qwen_counsel.py` directly in an srun session

### Can't find logs

- For sbatch: Check `logs/train_JOBID.out` and `logs/train_JOBID.err`
- For nohup: Check `nohup.out` or the file you specified
- For tmux/screen: Reattach to see output

### Job not using all GPUs

- Check with `nvidia-smi` in another terminal
- Verify `CUDA_VISIBLE_DEVICES` is set correctly
- Check training logs for GPU usage

### Want to monitor training progress

**For sbatch jobs (real-time output, but can't interact):**

```bash
# Method 1: Use helper script
./scripts/monitor_job.sh JOBID

# Method 2: Manual tail
tail -f logs/train_JOBID.out

# Method 3: Watch both output and errors
tail -f logs/train_JOBID.out logs/train_JOBID.err
```

**For srun + tmux (interactive, can "switch in"):**

```bash
# Reattach to tmux session
tmux attach -t qwen_training
# or
screen -r qwen_training
```

**Quick comparison:**

- `sbatch`: Watch output ✅ | Switch into job ❌
- `srun + tmux`: Watch output ✅ | Switch into job ✅
