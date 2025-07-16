"""
Script to run MPS compression benchmarks for multiple datasets.
"""

import numpy as np
from imgcompressionmps.evaluation.benchmark import run_full_benchmark


# --- Benchmark Configs ---

benchmark_configs= []

benchmark_configs += [
    {
        "name": f"fMRI_{start}_{start+5}_DCT",
        "dataset_path": "Data/fMRI_Datatset",
        "cutoff": np.linspace(0, 0.1, 100)[1:],
        "filename": f"ds000011_{start}_{start+5}_100steps_to_0p1_DCT.json",
        "datatype": "fMRI",
        "mode": "DCT",
        "start": start,
        "end": start + 5,
        "ending": ".gz",
        "shape": None
    }
    for start in range(15, 86, 5)  # adjust the upper bound as needed
]

"""
benchmark_configs += [
    {
        "name": f"Video - Pedestrians_{start}_{start+5}_Std",
        "dataset_path": "Data/pedestrians",
        "cutoff": np.linspace(0, 0.1, 100)[1:],
        "filename": f"Pedestrians_{start}_{start+5}_100steps_to_0p1_Std.json",
        "datatype": "Video",
        "mode": "Std",
        "start": start,
        "end": start + 5,
        "ending": ".npz",
        "shape": (200, 144, 216)
    }
    for start in range(155, 168, 5)
]


{
        "name": "Video - Pedestrians",
        "dataset_path": "Data/pedestrians",
        "cutoff": np.linspace(0, 0.1, 100)[1:],
        "filename": "Pedestrians_15_20_100steps_to_0p1_Std.json",
        "datatype": "Video",
        "mode": "Std",
        "start": 15,
        "end": 20,
        "ending": ".npz",
        "shape": (200, 144, 216)
    }
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