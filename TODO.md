# DeepLabv2 - Recap

## Cosa dice il Paper2?

### 🔧 **Ottimizzazione e training**

| Parametro                | Valore / Descrizione                                                                                                     |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| **Optimizer**            | Stochastic Gradient Descent (SGD)                                                                                        |
| **Learning rate**        | Non specificato per Cityscapes, ma su PASCAL VOC usano `0.001` e `0.01` per il classificatore finale. Si presume simile. |
| **Learning rate policy** | **poly** policy: `lr = initial_lr * (1 - iter / max_iter)^power`, con `power = 0.9`                                      |
| **Momentum**             | `0.9`                                                                                                                    |
| **Weight decay**         | `0.0005`                                                                                                                 |
| **Batch size**           | Non indicato specificamente per Cityscapes. Su PASCAL VOC, usano `10`                                                    |
| **Numero di iterazioni** | Non esplicitato, ma in altri esperimenti (e.g., PASCAL VOC) usano 20k                                                    |
| **Loss function**        | Somma delle cross-entropie sui pixel (con ground truth subsampled di 8×)                                                 |

## Prove che voglio fare
### Learning rate 
1. 0.0001
2. 0.0002
3. 0.0003

### Loss
1. CrossEntropy
2. Lovász
3. Tversky
4. Dice
5. Focal


### Optimizer
1. Stochastic Gradient Descent

❌ Sconsigliato cambiare ottimizzatore a metà training: DeepLabv2 + ResNet-101 è già molto sensibile alle modifiche dell’ottimizzazione. Cambiare da SGD a un altro ottimizzatore (es. Adam, RMSProp) distruggerebbe il regime di aggiornamento del learning rate (specie col poly). Si potrebbero introdurre salti instabili nella loss e ridurre la generalizzazione sul test set.

### Prove totali che voglio fare

| Prova | Loss Function | combinazione | mIoU         | Esecuzione |
| ----- | ------------- | -------------- | ------------ | ----- |
| 1     | CE            | —              | 50.50063     | FATTO |
| 2     | CE + Lovasz   | 0.5 / 0.5      | 50.82248     | FATTO |
| 3     | CE + Tversky  | 0.7 / 0.3      | 50.10477     | FATTO |
| 4     | CE + Dice     | 0.7 / 0.3      | 50.16859     | FATTO |


| Prova | Loss                | combinazione      | LR     | Extra Strategie  | mIoU         | Esecuzione |
| ----- | ------------------- | ----------------- | ------ | ---------------- | -------------| ---------- |
| 5     | CE + Lovasz         | 0.7 / 0.3         | 0.0002 | Warmup 500 step  | 53.79201     | FATTO      |
| 6     | CE + Focal          | γ=2.0             | 0.0003 | Warmup 500       | 53.9948      | FATTO      |
| 7     | CE + Lovasz + Dice  | 0.5 / 0.25 / 0.25 | 0.0002 | —                | 53.30673     | FATTO      |
| 8     | CE + Focal + Lovasz | 0.6 / 0.2 / 0.2   | 0.0001 | —                | 50.37082     | FATTO      |

**Aggiustamenti nei run**. 

Nel **run 5**, il più promettende, dopo 28 epoche ho preso la decisione di alzare il power a 1.2. Ecco le motivazioni:

Con il nuovo power = 1.2:

* Il learning rate scenderà un po’ più rapidamente,
* Il modello dovrebbe essere più stabile e rifinire meglio la segmentazione,
* E potenzialmente la mIoU potrà crescere già da questa epoca.

Dall'epoca 46 ho fatto queste altre modifiche
* alpha = 0.5, beta = 0.5 --> per stabilizzare (aggiungi motivazioni)
* freeze per layer = 1, 2, 3, 4 e bn1 del backbone --> per apprendere meglio (aggiungi motivazioni)
* power = 2 --> per fare decadere più velocemente il lr

All'epoca 47 ho modificato il power a 1.2 (discesa troppo rapida altrimenti)

Nel **run 6** ho cambiato il power → 1.0 per accelerare leggermente la discesa del learning rate e aiutare il modello a "chiudere meglio" nelle ultime 20 epoche (si spera).
All'epoca 44 ho freezato i layer 1, 2, 3, 4, bin1.

Nel **run 7** ho portato il power da 0.9 → 1.1 che mi ha permesso di:

* Raffinare più in fretta
* Ridurre un po’ di più il LR nelle prossime epoche
* Aiutare il modello a consolidare i dettagli di classe (soprattutto con DeepLabV2)


## Commentare i risultati 
MANCA 
C'è un notebook sul main di Auro già un po' fatto. Alla fine di questi 8 run, farei dei bei plot, loss e mIoU riassuntivi.

DeepLabV2_ce07_l03_warnup_lr_0.0002