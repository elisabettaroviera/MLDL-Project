import wandb
import os
from pathlib import Path

os.environ["WANDB_API_KEY"] = "a99b701251206c5e11379d8f1674d044e02abf5d"

api = wandb.Api()

entity = "s325951-politecnico-di-torino-mldl"
project = "3b_GTA5_to_CITY_augmented_color_3_random_tranforms_color_100_percent" # QUA METTI NOME PROGETTO


output_dir = Path(r"C:\Users\auron\OneDrive - Politecnico di Torino\Desktop\università\machine learning & deep learning\progetto\artifacts_and_csv")

output_dir.mkdir(parents=True, exist_ok=True)

runs = api.runs(f"{entity}/{project}")

for run in runs:
    print(f"📦 Run: {run.name}")
    for artifact in run.logged_artifacts():
        print(f"  🔍 Artifact: {artifact.name}")
        if "model_epoch_" in artifact.name:
            # Pulizia del nome per evitare caratteri non validi su Windows
            safe_artifact_name = artifact.name.replace(":", "_")
            dir_path = output_dir / f"{run.name}_{safe_artifact_name}"
            print(f"  ↓ Scarico artifact in {dir_path}")
            artifact.download(root=str(dir_path))
