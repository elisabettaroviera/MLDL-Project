# PROGRESS STATUS

## QUESTION

- Data augmentation va fatta sostituendo o aggiungendo le foto?


## DOVE SIAMO ARRIVAI...?
### 1st STEP. RELATED WORKS 
DONE

### 2nd STEP. TESTING SEMANTIC SEGMENTATION NETWORKS
#### 2.a Classic semantic segmentation network
**NN**. DeeplabV2 
**Dataset**. Cityscapes 
**Training epochs**. 50 
**Training resolution (Cityscapes)**. 1024x512 
**Test resolution (Cityscapes)**. 1024x512 
**Backbone**. R101 (pre-trained on ImageNet)

**Iperparametri**:
- batch_size = 3 # 3 or the number the we will use in the model
- learning_rate = 0.0001 # Learning rate for the optimizer - 1e-4
- momentum = 0.9 # Momentum for the optimizer
- weight_decay = 0.0005 # Weight decay for the optimizer

**Politiche di update lr**. Ogni batch

**Loss utilizzate**:
- CrossEntropy
- Tversky
- Lovász
- Dice

##### Run eseguite
1. CrossEntropy
2. 0.5 * CrossEntropy + 0.5 * Lovász
3. 0.7 * CrossEntropy + 0.3 * Tversky
4. 0.7 * CrossEntropy + 0.3 * Dice



#### 2.b Real-time semantic segmentation network.
**NN**. Bisenet 
**Dataset**. Cityscapes 
**Training epochs**. 50 
**Training resolution (Cityscapes)**. 1024x512 
**Test resolution (Cityscapes)**. 1024x512 
**Backbone**. ResNet18 (pre-trained on ImageNet)

**Iperparametri**:
- batch_size = 4 
- learning_rate = 0.00625
- momentum = 0.9
- weight_decay = 1e-4 

**Politiche di update lr**. Ogni batch o ogni epoca a seconda della run

**Loss utilizzate**:
- CrossEntropy
- Tversky
- Lovász
- Dice

##### Run eseguite
1. **LUCI** CrossEntropy

**Su 1/4 del dataset con lr=0.00625 con lr update tutti i batch**
- CrossEntropy
- Tversky
- Lovász
- Dice

****