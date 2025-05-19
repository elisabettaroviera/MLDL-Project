import wandb

# Configurazione del tuo progetto wandb
entity = "s328422-politecnico-di-torino"
project = "3b_GTA5_to_CITY_augmented_weather_cv07_tv_03"

api = wandb.Api()

# Recupera tutte le run del progetto
runs = api.runs(f"{entity}/{project}")

# Funzione per estrarre il numero dell'epoca dal nome della run
def extract_epoch_number(run):
    try:
        name = run.name
        if name.startswith("epoch_"):
            return int(name.split("_")[1])
    except:
        return float("inf")
    return float("inf")

# Filtra e ordina le run per numero di epoca
sorted_runs = sorted(
    [run for run in runs if run.name and run.name.startswith("epoch_")],
    key=extract_epoch_number
)

# Crea la lista degli ID delle run ordinate
run_ids = [run.id for run in sorted_runs]

# Ora puoi usare run_ids come vuoi, ad esempio:
print("Ho caricato", len(run_ids), "run ID.")
# Esempio: passare run_ids a una funzione
# validate_models(run_ids)