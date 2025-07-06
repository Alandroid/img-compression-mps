#!/bin/bash

# Path to your original Python script
script_path="cluster_benchmark.py"

# Loop over desired intervals
for ((start=0; start<5; start+=5)); do
    end=$((start+5))
    range_str="${start}_${end}"

    # Modify the line in-place using sed
    sed -i \
        -e "s/'ds003799_.*_Std_ses2.json'/'ds003799_${range_str}_100_100steps_to_0p1_Std_ses2.json'/" \
        -e "s/\"Std\", [0-9]\+, [0-9]\+, \".gz\"/\"Std\", ${start}, ${end}, \".gz\"/" \
        "$script_path"

    echo "Running benchmark for range: ${start}-${end}"
    ./../../../../max_run.sh
done

