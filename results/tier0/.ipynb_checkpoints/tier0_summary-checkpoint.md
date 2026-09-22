# Tier 0 re-analysis

Metric: Test Dice. Collapse threshold: held-out fold recall below 0.05.

## Fold mean vs ensemble

```
 Dataset        Model     Variant  Label Fraction (%)  Folds run  Fold mean Test Dice  Fold std Test Dice  Fold median Test Dice  Collapsed folds  bimodal  Ensemble Test Dice (filtered)  Folds combined (filtered)  Ensemble Test Dice (unfiltered)  Folds combined (unfiltered)
ISIC2018      BT-UNet      aug025                   5          5               0.5915              0.0235                 0.5912                0    False                         0.6090                     5.0000                           0.6090                            5
ISIC2018      BT-UNet      aug100                   5          5               0.6459              0.1470                 0.5541                0    False                         0.6962                     5.0000                           0.6962                            5
ISIC2018      BT-UNet        base                   1          5               0.5511              0.0217                 0.5390                0    False                         0.5658                     5.0000                           0.5658                            5
ISIC2018      BT-UNet        base                   5          5               0.5690              0.0474                 0.5574                0    False                         0.5657                     5.0000                           0.5657                            5
ISIC2018      BT-UNet        base                   5          5               0.5690              0.0474                 0.5574                0    False                         0.5657                     5.0000                           0.5657                            5
ISIC2018      BT-UNet        base                  10          5               0.7495              0.1271                 0.7950                0    False                         0.8117                     5.0000                           0.8117                            5
ISIC2018      BT-UNet        base                  10          5               0.7495              0.1271                 0.7950                0    False                         0.8117                     5.0000                           0.8117                            5
ISIC2018      BT-UNet        base                  20          5               0.8304              0.0041                 0.8309                0    False                         0.8435                     5.0000                           0.8435                            5
ISIC2018      BT-UNet        base                  20          5               0.8304              0.0041                 0.8309                0    False                         0.8435                     5.0000                           0.8435                            5
ISIC2018      BT-UNet        base                  50          5               0.8478              0.0059                 0.8477                0    False                         0.8606                     5.0000                           0.8606                            5
ISIC2018      BT-UNet        base                  50          5               0.8478              0.0059                 0.8477                0    False                         0.8606                     5.0000                           0.8606                            5
ISIC2018      BT-UNet        bs32                   5          5               0.6236              0.1158                 0.5667                0    False                         0.6026                     5.0000                           0.6026                            5
ISIC2018      BT-UNet        bs64                   5          5               0.6662              0.1261                 0.5847                0    False                         0.6519                     5.0000                           0.6519                            5
ISIC2018      BT-UNet    cropflip                   5          5               0.5780              0.0192                 0.5866                0    False                         0.5900                     5.0000                           0.5900                            5
ISIC2018      BT-UNet          e0                   5          5               0.0221              0.0491                 0.0001                5    False                            NaN                        NaN                           0.0001                            5
ISIC2018      BT-UNet        e200                   5          5               0.6076              0.0171                 0.6110                0    False                         0.6157                     5.0000                           0.6157                            5
ISIC2018      BT-UNet         e25                   5          5               0.5537              0.0195                 0.5524                0    False                         0.5750                     5.0000                           0.5750                            5
ISIC2018      BT-UNet         e50                   5          5               0.5453              0.0094                 0.5460                0    False                         0.5749                     5.0000                           0.5749                            5
ISIC2018      BT-UNet      frozen                   1          5               0.5701              0.0514                 0.5890                0    False                         0.5893                     5.0000                           0.5893                            5
ISIC2018      BT-UNet      frozen                   5          5               0.5674              0.0247                 0.5630                0    False                         0.5897                     5.0000                           0.5897                            5
ISIC2018      BT-UNet      frozen                  10          5               0.6649              0.0695                 0.6272                0    False                         0.6978                     5.0000                           0.6978                            5
ISIC2018      BT-UNet  lambda0001                   5          5               0.5816              0.0404                 0.6054                0    False                         0.5950                     5.0000                           0.5950                            5
ISIC2018      BT-UNet   lambda002                   5          5               0.5594              0.0280                 0.5537                0    False                         0.5609                     5.0000                           0.5609                            5
ISIC2018      BT-UNet     lossbce                   5          5               0.6109              0.0108                 0.6055                0    False                         0.6276                     5.0000                           0.6276                            5
ISIC2018      BT-UNet    lossdice                   5          5               0.6060              0.0294                 0.6177                0    False                         0.6220                     5.0000                           0.6220                            5
ISIC2018      BT-UNet    proj2048                   5          5               0.5504              0.1201                 0.5298                0    False                         0.5625                     5.0000                           0.5625                            5
ISIC2018      BT-UNet     proj256                   5          5               0.5886              0.0155                 0.5903                0    False                         0.6131                     5.0000                           0.6131                            5
ISIC2018      BT-UNet     proj512                   5          5               0.5842              0.0411                 0.5828                0    False                         0.5896                     5.0000                           0.5896                            5
ISIC2018      BT-UNet       smoke                   1          5               0.5496              0.0339                 0.5562                0    False                         0.5654                     5.0000                           0.5654                            5
ISIC2018      BT-UNet sslbudget20                  20          5               0.8312              0.0060                 0.8342                0    False                         0.8437                     5.0000                           0.8437                            5
ISIC2018      BT-UNet  sslbudget5                   5          5               0.0001              0.0000                 0.0001                5    False                            NaN                        NaN                           0.0001                            5
ISIC2018 SimSiam-UNET      aug025                   5          5               0.6521              0.0805                 0.6179                0    False                         0.6414                     5.0000                           0.6414                            5
ISIC2018 SimSiam-UNET      aug100                   5          5               0.6630              0.1162                 0.6038                0    False                         0.6739                     5.0000                           0.6739                            5
ISIC2018 SimSiam-UNET        base                   1          5               0.5304              0.0544                 0.5162                0    False                         0.5372                     5.0000                           0.5372                            5
ISIC2018 SimSiam-UNET        base                   5          5               0.6311              0.1001                 0.5954                0    False                         0.6233                     5.0000                           0.6233                            5
ISIC2018 SimSiam-UNET        base                  10          5               0.7327              0.1231                 0.8083                0    False                         0.8160                     5.0000                           0.8160                            5
ISIC2018 SimSiam-UNET        base                  20          5               0.8402              0.0036                 0.8414                0    False                         0.8498                     5.0000                           0.8498                            5
ISIC2018 SimSiam-UNET        base                  50          5               0.8524              0.0043                 0.8507                0    False                         0.8630                     5.0000                           0.8630                            5
ISIC2018 SimSiam-UNET        bs32                   5          5               0.6223              0.0954                 0.5776                0    False                         0.6201                     5.0000                           0.6201                            5
ISIC2018 SimSiam-UNET        bs64                   5          5               0.6055              0.0376                 0.5914                0    False                         0.6050                     5.0000                           0.6050                            5
ISIC2018 SimSiam-UNET    cropflip                   5          5               0.6145              0.0745                 0.5878                0    False                         0.6000                     5.0000                           0.6000                            5
ISIC2018 SimSiam-UNET          e0                   5          5               0.3560              0.2408                 0.4140                1     True                         0.3905                     4.0000                           0.2950                            5
ISIC2018 SimSiam-UNET        e200                   5          5               0.6105              0.0933                 0.5961                0    False                         0.6251                     5.0000                           0.6251                            5
ISIC2018 SimSiam-UNET         e25                   5          5               0.6224              0.1068                 0.5864                0    False                         0.6187                     5.0000                           0.6187                            5
ISIC2018 SimSiam-UNET         e50                   5          5               0.6291              0.0897                 0.6007                0    False                         0.6295                     5.0000                           0.6295                            5
ISIC2018 SimSiam-UNET      frozen                   1          5               0.5620              0.0139                 0.5623                0    False                         0.5765                     5.0000                           0.5765                            5
ISIC2018 SimSiam-UNET      frozen                   5          5               0.5894              0.0165                 0.5946                0    False                         0.6142                     5.0000                           0.6142                            5
ISIC2018 SimSiam-UNET      frozen                  10          5               0.6437              0.0717                 0.5998                0    False                         0.6585                     5.0000                           0.6585                            5
ISIC2018 SimSiam-UNET  nostopgrad                   5          5               0.6257              0.0946                 0.5965                0    False                         0.6284                     5.0000                           0.6284                            5
ISIC2018 SimSiam-UNET     pred128                   5          5               0.5672              0.0242                 0.5769                0    False                         0.5883                     5.0000                           0.5883                            5
ISIC2018 SimSiam-UNET      pred32                   5          5               0.5833              0.0223                 0.5863                0    False                         0.6004                     5.0000                           0.6004                            5
ISIC2018 SimSiam-UNET    proj2048                   5          5               0.6434              0.0916                 0.5944                0    False                         0.6399                     5.0000                           0.6399                            5
ISIC2018 SimSiam-UNET     proj256                   5          5               0.5923              0.0339                 0.5778                0    False                         0.5938                     5.0000                           0.5938                            5
ISIC2018 SimSiam-UNET     proj512                   5          5               0.6379              0.0993                 0.6080                0    False                         0.6361                     5.0000                           0.6361                            5
ISIC2018 SimSiam-UNET sslbudget20                  20          5               0.8334              0.0068                 0.8347                0    False                         0.8450                     5.0000                           0.8450                            5
ISIC2018 SimSiam-UNET  sslbudget5                   5          5               0.6234              0.0885                 0.5962                0    False                         0.6102                     5.0000                           0.6102                            5
ISIC2018         UNet        base                   1          5               0.3220              0.2983                 0.4511                2     True                         0.5840                     3.0000                           0.2321                            5
ISIC2018         UNet        base                   5          5               0.2336              0.2927                 0.0627                3     True                         0.6116                     2.0000                           0.0193                            5
ISIC2018         UNet        base                  10          5               0.8018              0.0077                 0.7986                0    False                         0.8128                     5.0000                           0.8128                            5
ISIC2018         UNet        base                  20          5               0.8351              0.0038                 0.8353                0    False                         0.8476                     5.0000                           0.8476                            5
ISIC2018         UNet        base                  50          5               0.8496              0.0039                 0.8508                0    False                         0.8603                     5.0000                           0.8603                            5
ISIC2018         UNet     lossbce                   5          5               0.0867              0.1671                 0.0003                4    False                         0.3832                     1.0000                           0.0001                            5
ISIC2018         UNet    lossdice                   5          5               0.2158              0.2016                 0.2075                2     True                         0.3052                     3.0000                           0.0754                            5
```

## Cells with collapsed folds

Any ensemble in these cells was built from a subset of its folds, so it is not comparable with an ensemble in a cell where nothing was excluded.

```
 Dataset        Model    Variant  Label Fraction (%)  n  collapsed  collapse_rate   mean  median
ISIC2018      BT-UNet         e0                   5  5          5         1.0000 0.0221  0.0001
ISIC2018      BT-UNet sslbudget5                   5  5          5         1.0000 0.0001  0.0001
ISIC2018 SimSiam-UNET         e0                   5  5          1         0.2000 0.3560  0.4140
ISIC2018         UNet       base                   1  5          2         0.4000 0.3220  0.4511
ISIC2018         UNet       base                   5  5          3         0.6000 0.2336  0.0627
ISIC2018         UNet    lossbce                   5  5          4         0.8000 0.0867  0.0003
ISIC2018         UNet   lossdice                   5  5          2         0.4000 0.2158  0.2075
```

## Cells where the mean is misleading

Folds split into working and failed groups with nothing in between. Report the collapse rate and the median; the mean falls in the gap.

```
 Dataset        Model  Variant  Label Fraction (%)  n   mean  median    min    max
ISIC2018 SimSiam-UNET       e0                   5  5 0.3560  0.4140 0.0879 0.5817
ISIC2018         UNet     base                   1  5 0.3220  0.4511 0.0001 0.5828
ISIC2018         UNet     base                   5  5 0.2336  0.0627 0.0001 0.5765
ISIC2018         UNet lossdice                   5  5 0.2158  0.2075 0.0001 0.5165
```
