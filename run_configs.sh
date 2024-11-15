#!/bin/bash

# Define your configurations
for hid_dim in 128 256; do
  for pool_ratio in 0.6 0.7 0.8; do
    python main.py --hid_dim $hid_dim --pool_ratio $pool_ratio
  done
done
