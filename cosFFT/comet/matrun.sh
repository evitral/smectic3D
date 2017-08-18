#!/bin/bash
# Use debug partition for MATLAB

srun --partition=debug --pty --nodes=1 --ntasks-per-node=1 -t 00:05:00 --wait=0 --export=ALL /bin/bash
