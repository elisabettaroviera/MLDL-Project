import torch
import torch.nn as nn

affine_par = True

# ----------------------
# Classe Bottleneck
# ----------------------
# Questa classe rappresenta un blocco residuo ("bottleneck") usato nella ResNet.
# Esegue tre convoluzioni (1x1 → 3x3 → 1x1) e aggiunge una connessione residua (skip connection).
# Se necessario, fa il "downsample" del residuo per farlo combaciare.
# Usata per costruire i layer1–4 della ResNet.

# **Bottleneck**. This class implements a modified version of the standard ResNet bottleneck residual block.
# It includes support for optional dilation and handling the downsampling connection. A key feature is that
# its batch normalization layers are configured to have their parameters fixed (not trained).
class Bottleneck(nn.Module):
    # **Bottleneck.expansion**. This is a class attribute. It indicates the channel expansion factor
    # within the bottleneck block, which is set to 4.
    expansion = 4

    # **Bottleneck.__init__**. This is the constructor method. It initializes the three convolutional
    # layers (`conv1`, `conv2`, `conv3`), their corresponding batch normalization layers
    # (`bn1`, `bn2`, `bn3`), and a ReLU activation. It explicitly sets `requires_grad=False`
    # for the parameters of all three batch normalization layers. It also takes parameters for
    # `stride`, `dilation`, and an optional `downsample` module.
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    # **Bottleneck.forward**. This method defines the forward pass for the bottleneck block.
    # It processes the input (`x`) sequentially through the three convolutional layers with
    # intermediate BN and ReLU activations. It computes the residual connection (applying the
    # `downsample` module to the input if provided) and adds this `residual` to the main path
    # output (`out`) before the final ReLU activation.
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

# ----------------------
# Classe ClassifierModule
# ----------------------
# È il modulo ASPP (Atrous Spatial Pyramid Pooling) in DeepLab v2.
# Contiene convoluzioni 3x3 con diversi dilation e padding.
# Somma le predizioni di tutti i rami per ottenere una mappa finale.
# In ResNetMulti, è chiamato layer6.

# **ClassifierModule**. This module implements a simple classification head, likely used for pixel-wise prediction
# in a semantic segmentation model. It applies multiple 3x3 convolutional layers to the input feature map, each using
# a different dilation rate from a provided series (`dilation_series`), and then sums the outputs of these convolutions.
class ClassifierModule(nn.Module):
    # **ClassifierModule.__init__**. This is the constructor method. It initializes a `nn.ModuleList` 
    # to hold multiple `nn.Conv2d` layers. It iterates through the provided `dilation_series` and 
    # `padding_series` to create a 3x3 convolutional layer for each pair, setting the corresponding 
    # `dilation` and `padding`. It also initializes the weights of these convolutional layers using 
    # a normal distribution.
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(ClassifierModule, self).__init__()
        self.conv2d_list = nn.ModuleList() #credo siano i 4 branch  in ASPP diciamo cioè quelli che poi vengono sommati
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding,
                          dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    # **ClassifierModule.forward**. This method defines the forward pass. It applies each convolutional 
    # layer within `self.conv2d_list` to the input feature map `x` and returns the sum of their outputs.
    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x) #infatti qua li sommo (sum-fusion in figura pag 8)
        return out

# ----------------------
# Classe ResNetMulti
# ----------------------
# È il backbone completo ResNet + ASPP, adattato per la segmentazione semantica.
# Struttura:
#   - conv1 + bn1 + ReLU + maxpool
#   - layer1: 3 bottleneck
#   - layer2: 4 bottleneck
#   - layer3: 23 bottleneck (con dilation=2)
#   - layer4: 3 bottleneck (con dilation=4)
#   - layer6: ClassifierModule (ASPP)
# Nel forward, restituisce la predizione finale (upscalata alla dimensione originale dell’immagine).

# **ResNetMulti**. This class implements a modified ResNet backbone, specifically designed for tasks like 
# semantic segmentation (similar to the backbone used in DeepLabV2). It utilizes the `Bottleneck` blocks, 
# incorporates dilated convolutions in the later layers (`layer3`, `layer4`) to preserve spatial resolution, 
# and includes the `ClassifierModule` as the final classification head (`layer6`). The parameters of batch normalization
# layers within the main body are fixed.
class ResNetMulti(nn.Module):
    # **ResNetMulti.__init__**. This is the constructor method. It initializes the initial layers of
    # the network (conv1, bn1, relu, maxpool) and builds the main sequential layers (`layer1` through `layer4`)
    # using the `_make_layer` helper method, applying specific strides and dilation rates (dilation 2 for 
    # layer3, dilation 4 for layer4). It also initializes `layer6` using the `ClassifierModule` and applies 
    # weight initialization to most modules within the network.
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNetMulti, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer6 = ClassifierModule(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # ----------------------
    # Metodo _make_layer
    # ----------------------
    # Costruisce una sequenza di blocchi Bottleneck (es. 3, 4, 6, 3 per ResNet-50).
    # Decide se applicare downsampling (es. per cambiare la dimensione dei canali o la risoluzione).
    # Crea i layer1, layer2, layer3 e layer4 nella classe ResNetMulti.            

    # **ResNetMulti._make_layer**. This is a helper method used by the `__init__` method to construct a 
    # sequence of `Bottleneck` blocks. It handles the creation of the `downsample` module (a 1x1 convolution 
    # followed by Batch Normalization) when it's needed (due to changes in feature map dimensions or the 
    # presence of dilation). It specifically sets `requires_grad=False` for the batch normalization 
    # parameters within the downsample module. It then creates and appends the specified number of 
    # `Bottleneck` blocks to a list, returning them as a `nn.Sequential` module.
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if (stride != 1
                or self.inplanes != planes * block.expansion
                or dilation == 2
                or dilation == 4):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    
    # **ResNetMulti.forward**. This method defines the forward pass of the entire `ResNetMulti` model. 
    # It processes the input through the initial layers, then through `layer1` to `layer4`. 
    # The output of `layer4` is then passed to the `ClassifierModule` (`layer6`). The final 
    # output of `layer6` is bilinearly upsampled back to the original input spatial dimensions. 
    # During training (`self.training == True`), it returns the upsampled output along with two `None` values.
    def forward(self, x):
        _, _, H, W = x.size()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer6(x)

        x = torch.nn.functional.interpolate(x, size=(H, W), mode='bilinear')

        if self.training == True:
            return x, None, None

        return x

    # **ResNetMulti.get_1x_lr_params_no_scale**. This is a generator method intended to yield parameters 
    # from most of the network's layers (conv1, bn1, layer1-layer4), typically used to set up an optimizer. 
    # It specifically excludes parameters where `requires_grad` is False (such as the fixed batch 
    # normalization parameters) and parameters from the final classification layer (`layer6`).
    def get_1x_lr_params_no_scale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    # **ResNetMulti.get_10x_lr_params**. This is a generator method intended to yield parameters specifically
    #  from the final classification layer (`layer6`). This is often used to assign a higher learning rate
    #  (e.g., 10 times the base rate) to the classifier in the optimizer, as it's trained from scratch 
    # while the backbone is pre-trained.
    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        if self.multi_level:
            b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    # **ResNetMulti.optim_parameters**. This method prepares parameter groups suitable for configuring 
    # an optimizer (like `torch.optim.SGD`). It returns a list containing two dictionaries: one with 
    # parameters from the main body (obtained via `get_1x_lr_params_no_scale`) assigned a base learning 
    # rate (`lr`), and one with parameters from the classifier (obtained via `get_10x_lr_params`) assigned 
    # a learning rate of `10 * lr`.
    def optim_parameters(self, lr):
        return [{'params': self.get_1x_lr_params_no_scale(), 'lr': lr},
                {'params': self.get_10x_lr_params(), 'lr': 10 * lr}]


# ----------------------
# Metodo get_deeplab_v2
# ----------------------
# Istanzia il modello ResNetMulti con configurazione DeepLab v2.
# Usa la struttura ResNet-101 (3, 4, 23, 3 bottleneck blocks).
# Carica i pesi pre-addestrati da un checkpoint di ImageNet, se pretrain=True.

# **get_deeplab_v2**. This function serves as a factory to create an instance of the `ResNetMulti`
# model specifically configured as a DeepLabV2 backbone (using the `Bottleneck` block and layer 
# counts corresponding to a ResNet-101). It also includes functionality to load pre-trained weights 
# from a specified file path if the `pretrain` flag is set to `True`, adapting the state dictionary 
# keys as needed.
def get_deeplab_v2(num_classes=19, pretrain=True, pretrain_model_path='DeepLab_resnet_pretrained_imagenet.pth'):
    model = ResNetMulti(Bottleneck, [3, 4, 23, 3], num_classes)

    # Pretraining loading
    if pretrain:
        print('Deeplab pretraining loading...')
        saved_state_dict = torch.load(pretrain_model_path)

        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        model.load_state_dict(new_params, strict=False)

    return model

# Learning rate policy function
# optimizer (torch.optim.Optimizer): Optimizer to update.
# init_lr (float): Initial learning rate
# iter (int): Current iteration.
# max_iter (int): Maximum number of iterations.
# power (float): Power factor (default = 0.9).
def lr_policy(optimizer, init_lr, iter, max_iter, power=0.9):
    new_lr = init_lr * (1 - iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr # Update the learning rate in the optimizer
    return new_lr
