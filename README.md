# LabelCraft
[![arXiv](https://img.shields.io/badge/arXiv-2502.09992-red.svg)](https://arxiv.org/abs/2312.10947)
This is the pytorch implementation of our paper at WSDM 2024:
> [LabelCraft: Empowering Short Video Recommendations with Automated Label Crafting](https://arxiv.org/abs/2312.10947)
> 
> Yimeng Bai, Yang Zhang, Jing Lu, Jianxin Chang, Xiaoxue Zang, Yanan Niu, Yang Song, Fuli Feng.

## Usage
### Data
The experimental datasets Kuaishou and WeChat are available for download via the links provided in the files located at `/data/kuaishou/download.txt` and `/data/wechat/download.txt`.
### Training & Evaluation
```
python main.py
```
## Citation
```
@inproceedings{LabelCraft,
author = {Bai, Yimeng and Zhang, Yang and Lu, Jing and Chang, Jianxin and Zang, Xiaoxue and Niu, Yanan and Song, Yang and Feng, Fuli},
title = {LabelCraft: Empowering Short Video Recommendations with Automated Label Crafting},
year = {2024},
isbn = {9798400703713},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3616855.3635816},
doi = {10.1145/3616855.3635816},
booktitle = {Proceedings of the 17th ACM International Conference on Web Search and Data Mining},
pages = {28–37},
numpages = {10},
keywords = {label generation, short video recommendation},
location = {Merida, Mexico},
series = {WSDM '24}
}
```
