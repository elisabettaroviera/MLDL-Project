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


### Optimizer
1. Stochastic Gradient Descent

❌ Sconsigliato cambiare ottimizzatore a metà training, DeepLabv2 + ResNet-101 è già molto sensibile alle modifiche dell’ottimizzazione. Cambiare da SGD a un altro ottimizzatore (es. Adam, RMSProp) distruggerebbe il regime di aggiornamento del learning rate (specie col poly). Si potrebbero introdurre salti instabili nella loss e ridurre la generalizzazione sul test set.

### Prove totali che voglio fare

| Prova | Loss Function | λ combinazione | mIoU         |
| ----- | ------------- | -------------- | ------------ |
| 1     | CE            | —              | 50.50063     |
| 2     | CE + Lovasz   | 0.5 / 0.5      | 50.82248     |
| 3     | CE + Tversky  | 0.7 / 0.3      | 50.10477     |
| 4     | CE + Dice     | 0.7 / 0.3      | 50.16859     |


| Prova | Loss                | λ / gamma         | LR     | Extra Strategie  | mIoU         |
| ----- | ------------------- | ----------------- | ------ | ---------------- | -------------|
| 5     | CE + Lovasz         | 0.7 / 0.3         | 0.0002 | Warmup 500 step  |              |
| 6     | CE + Focal          | γ=2.0             | 0.0003 | Warmup 500 + TTA |              |
| 7     | CE + Lovasz + Dice  | 0.5 / 0.25 / 0.25 | 0.0002 | —                |              |
| 8     | CE + Focal + Lovasz | 0.6 / 0.2 / 0.2   | 0.0001 | —                |              |
| 9     | CE + Boundary Loss  | 0.5 / 0.5         | 0.0003 | Warmup + TTA     |              |
| 10    | Focal Loss only     | γ=2.0             | 0.0001 | —                |              |
| 11    | CE + Tversky        | 0.5 / 0.5         | 0.0002 | Warmup           |              |
| 12    | CE + Dice           | 0.5 / 0.5         | 0.0003 | —                |              |

