"""
Script to run MPS compression benchmarks for multiple datasets.
"""

import numpy as np
from imgcompressionmps.evaluation.benchmark import run_full_benchmark


# --- Benchmark Configs ---
benchmark_configs = [
    
    {
        "name": "Video - Pedestrians",
        "dataset_path": "Data/pedestrians",
        "cutoff": np.linspace(0, 0.1, 10)[1:],
        "filename": "Pedestrians_0_1_10steps_to_0p1_Std_new_new.json",
        "datatype": "Video",
        "mode": "Std",
        "start": 0,
        "end": 1,
        "ending": ".npz",
        "shape": (200, 144, 216)
    }
]

"""
{
        "name": "Video - Pedestrians",
        "dataset_path": "Data/pedestrians",
        "cutoff": np.linspace(0, 0.1, 100)[1:],
        "filename": "Pedestrians_0_10_100steps_to_0p1_Std.json",
        "datatype": "Video",
        "mode": "Std",
        "start": 0,
        "end": 10,
        "ending": ".npz",
        "shape": (200, 144, 216)
    },
{
        "name": "fMRI",
        "dataset_path": "Data/fMRI_Datatset",
        "cutoff": np.linspace(0, 0.1, 100)[1:],
        "filename": "fMRI_0_5_100steps_to_0p1_Std.json",
        "datatype": "fMRI",
        "mode": "Std",
        "start": 0,
        "end": 5,
        "ending": ".gz",
        "shape": None
    },
,
    
    {
        "name": "MRI Slice",
        "dataset_path": "Data/ds003799-2.0.0",
        "cutoff": np.linspace(0, 0.3, 100)[1:],
        "filename": "MRI_slice_ds003799_0_to_50_100_steps_to_03_Std.json",
        "datatype": "MRI_Slice",
        "mode": "Std",
        "start": 0,
        "end": 50,
        "ending": ".gz",
        "shape": None
    }



"""




# --- Run All Benchmarks ---
if __name__ == "__main__":
    for cfg in benchmark_configs:
        print(f"\n--- Running benchmark: {cfg['name']} ---")
        run_full_benchmark(
            dataset_path=cfg["dataset_path"],
            cutoff_list=cfg["cutoff"],
            result_file=cfg["filename"],
            datatype=cfg["datatype"],
            mode=cfg["mode"],
            start=cfg["start"],
            end=cfg["end"],
            ending=cfg["ending"],
            shape=cfg["shape"]
        )