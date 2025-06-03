import wandb
import os
from pathlib import Path
import re
from collections import defaultdict

# API Key
os.environ["WANDB_API_KEY"] = "a99b701251206c5e11379d8f1674d044e02abf5d"
api = wandb.Api()

entity = "s325951-politecnico-di-torino-mldl"
project = "3b_GTA5_to_CITY_augmented_color_2_random_tranform_color_OR_ALL_g_h_i_100_percent"
#3b_GTA5_to_CITY_augmented_color_2_random_tranforms_color_+_1_from_g_h_i_100_percent0
#3b_GTA5_to_CITY_augmented_color_1_random_tranform_color_plus_1_from_g_h_i_100_percent

# Percorso locale di download
output_dir = Path(r"C:\Users\auron\OneDrive - Politecnico di Torino\Desktop\università\machine learning & deep learning\progetto\artifacts_and_csv\3b_GTA5_to_CITY_augmented_color_2_random_tranform_color_OR_ALL_g_h_i_100_percent")
output_dir.mkdir(parents=True, exist_ok=True)

# Raggruppamento per run + epoca
artifact_dict = defaultdict(list)

runs = api.runs(f"{entity}/{project}")

for run in runs:
    print(f"\n📦 Run: {run.name}")
    for artifact in run.logged_artifacts():
        name = artifact.name  # es: model_epoch_20
        if name.startswith("model_epoch_"):
            match = re.search(r"model_epoch_(\d+)", name)
            if match:
                epoch = int(match.group(1))
                key = (run.name, epoch)
                artifact_dict[key].append(artifact)
                print(f"  ➕ Trovato: {name} (v{artifact.version})")

# Funzione per ordinare per versione (es: v0, v1, ...)
def get_version_number(artifact):
    match = re.search(r"v(\d+)", artifact.version)
    return int(match.group(1)) if match else -1

# Scarica l’ultima versione per ogni epoca
for (run_name, epoch), artifacts in artifact_dict.items():
    latest_artifact = sorted(artifacts, key=get_version_number, reverse=True)[0]
    safe_artifact_name = latest_artifact.name.replace(":", "_")
    dir_path = output_dir / f"{run_name}_epoch_{epoch}"
    print(f"  ↓ Scarico {safe_artifact_name} (ultima versione) in {dir_path}")
    latest_artifact.download(root=str(dir_path))
