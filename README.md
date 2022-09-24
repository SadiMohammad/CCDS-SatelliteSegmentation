# CCDS-SatelliteSegmentation

## Training

```
cd CCDS-SatelliteSegmentation
mkdir pretrained
cd pretrained
wget https://download.pytorch.org/models/resnet50-19c8e357.pth # ResNet50
wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth # ResNet101
```
Train model with the following command

```
python3 train.py --config_file train
```