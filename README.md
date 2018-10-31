# Fine-Grained-Cloth-Classification
Beating Fine-Grained Cloth Classification benchmark using Fast.AI in 10 Lines of Code

```
from fastai import *
from fastai.vision import *
path = Path("data/cloth_categories/")
data = ImageDataBunch.from_csv(path, csv_labels="train_labels.csv", ds_tfms=get_transforms(), size=224)
data.normalize(imagenet_stats)
learn = create_cnn(data, models.resnet34, metrics=accuracy)
learn.fit_one_cycle(8)
learn.save('stage-1_sz-150')
```

This is repo for my article

https://medium.com/@pankajmathur/clothing-categories-classification-using-fast-ai-v1-0-in-10-lines-of-code-4e848797721

Make Sure to follow article to do below steps, before trying this notebook.

* Download dataset http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
* Installing & Setting up Fast.AI Libraries https://github.com/fastai/fastai



Huge Thanks to @Ziwei Liu, @Ping Luo, @Shi Qiu, @Xiaogang Wang, and @Xiaoou Tang from The Chinese University of Hong Kong

@inproceedings{liu2016deepfashion,
 author = {Ziwei Liu, Ping Luo, Shi Qiu, Xiaogang Wang, and Xiaoou Tang},
 title = {DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations},
 booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 month = June,
 year = {2016} 
 }
