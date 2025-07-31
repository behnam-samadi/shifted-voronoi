import os
import subprocess

# Define where to save the data
output_dir = "/home/samadi/research/pythonProject/ScanNetDataset/"

# Files you want per scan
file_types = [
    "_vh_clean_2.ply",
    ".aggregation.json",
    ".seg.json",
    ".txt"
]

# Paths to your scene ID list files
scene_files = [
    "/home/samadi/research/pythonProject/ScanNetDataset/scannetv2_train.txt",
    "/home/samadi/research/pythonProject/ScanNetDataset/scannetv2_test.txt"
]

# Read all scene IDs
scene_ids = []
for file in scene_files:
    with open(file, "r") as f:
        for line in f:
            scene_id = line.strip()
            if scene_id:
                scene_ids.append(scene_id)

# Download each scan individually
for scene_id in scene_ids:
    print(f"📥 Downloading {scene_id} ...")
    args = [
        "python", "/home/samadi/research/pythonProject/ScanNetDataset/download-scannet.py",
        "-o", output_dir,
        "--id", scene_id,
        "--v1",               # for ScanNetV1 file structure
        "--skip_existing"     # don't re-download existing files
    ]
    for t in file_types:
        args += ["--type", t]

    subprocess.run(args)

