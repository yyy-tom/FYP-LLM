#!/bin/bash
# Helper script to monitor SLURM batch jobs in real-time
# Usage: ./monitor_job.sh [JOBID]

if [ -z "$1" ]; then
    # No job ID provided, show running jobs and let user choose
    echo "Your running jobs:"
    squeue -u $USER -o "%.10i %.20j %.8T %.10M %.6D %R"
    echo ""
    echo "Usage: $0 JOBID"
    echo "Or provide job ID: $0 <job_id>"
    exit 1
fi

JOBID=$1

# Check if job exists
if ! squeue -j $JOBID &>/dev/null; then
    echo "Job $JOBID not found or not running."
    echo "Checking for log files..."
    
    # Try to find log files
    if [ -f "logs/train_${JOBID}.out" ]; then
        echo "Found log file: logs/train_${JOBID}.out"
        echo "Showing last 50 lines:"
        tail -50 logs/train_${JOBID}.out
    else
        echo "No log file found for job $JOBID"
    fi
    exit 1
fi

# Get job name
JOB_NAME=$(squeue -j $JOBID -h -o %j)

echo "=========================================="
echo "Monitoring Job: $JOBID ($JOB_NAME)"
echo "=========================================="
echo ""
echo "Press Ctrl+C to stop monitoring"
echo "Log files:"
echo "  Output: logs/train_${JOBID}.out"
echo "  Errors: logs/train_${JOBID}.err"
echo ""
echo "=========================================="
echo ""

# Monitor both output and error files
if [ -f "logs/train_${JOBID}.out" ] && [ -f "logs/train_${JOBID}.err" ]; then
    # Use multitail if available, otherwise tail -f
    if command -v multitail &> /dev/null; then
        multitail -s 2 logs/train_${JOBID}.out logs/train_${JOBID}.err
    else
        echo "=== OUTPUT ==="
        tail -f logs/train_${JOBID}.out &
        TAIL_PID=$!
        echo "=== ERRORS ==="
        tail -f logs/train_${JOBID}.err
        kill $TAIL_PID 2>/dev/null
    fi
elif [ -f "logs/train_${JOBID}.out" ]; then
    tail -f logs/train_${JOBID}.out
else
    echo "Waiting for log file to be created..."
    while [ ! -f "logs/train_${JOBID}.out" ]; do
        sleep 1
    done
    tail -f logs/train_${JOBID}.out
fi

