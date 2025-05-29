# PROSSIME RUN BISENET

| Rank | Loss Function                             | LR         | Warmup | Optimizer | Scheduler  | Note sintetica                                                                    |
| ---- | ----------------------------------------- | ---------- | ------ | --------- | ---------- | --------------------------------------------------------------------------------- |
| 1    | **CE + Focal (0.5/0.5)**                  | 0.000625   | ✅      | SGD       | Poly (0.9) | Ottimo compromesso stabilità/improvement. Funziona bene con class imbalance       |
| 2    | **CE + Focal + Tversky (0.5/0.3/0.2)**    | 0.000625   | ✅      | SGD       | Poly       | Combinazione più promettente in assoluto. Richiede tuning, ma molto forte         |
| 3    | **CE → Lovász** (ultime 15 ep)            | 0.000625   | ✅      | SGD       | Poly       | Loss dinamica: punta dritta a massimizzare mIoU dopo stabilizzazione iniziale     |
| 4    | **CE + Tversky (0.7/0.3)**                | 0.000625   | ✅      | SGD       | Poly       | Versione migliorata della tua migliore run. Con warmup può salire ancora          |
| 5    | **CE + Lovász (0.6/0.4)**                 | 0.000625   | ✅      | SGD       | Poly       | Target diretto su mIoU. Buona in combinazione col warmup                          |
| 6    | **Dice + Focal (0.5/0.5)**                | 0.000625   | ✅      | SGD       | Poly       | Approccio alternativo senza CE, ottimo per provare altre metriche differenziabili |
| 7    | **CE + Focal (0.5/0.5)**                  | **0.001**  | ✅      | SGD       | Poly       | Come la #1 ma con lr leggermente più alto → per testare sensibilità               |
| 8    | **CE + Tversky + Lovász (0.5/0.25/0.25)** | 0.000625   | ✅      | SGD       | Poly       | Variante della tua run 3 più equilibrata nei pesi                                 |
| 9    | **CE + Focal (0.5/0.5)**                  | 0.000625   | ✅      | **AdamW** | Poly       | Solo per testare se AdamW migliora convergenza con stessa loss                    |
| 10   | **CE + Focal (0.5/0.5)**                  | **0.0005** | ✅      | SGD       | Poly       | Run a lr più bassa per stabilità se vedi oscillazioni                             |
