# SoundNet
### The pytorch version of SoundNet
![Github](https://img.shields.io/badge/PyTorch-v0.4.1-red.svg?style=for-the-badge&logo=data:image/png)
![Github](https://img.shields.io/badge/python-3.5-green.svg?style=for-the-badge&logo=python)

![](https://camo.githubusercontent.com/0b88af5c13ba987a17dcf90cd58816cf8ef04554/687474703a2f2f70726f6a656374732e637361696c2e6d69742e6564752f736f756e646e65742f736f756e646e65742e6a7067)

Abstract
---
This repository arrange the structure of SoundNet. You can use the pre-trained model directly. This repository is revised from [here](https://github.com/EsamGhaleb/soundNet_pytorch).    

Usage
---
```python
model = SoundNet()
model.load_state_dict(torch.load('sound8.pth'))
```