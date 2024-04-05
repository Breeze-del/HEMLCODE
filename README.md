# HEML_CODE
## HEML
This is our implementation for our paper **Hypergraph-Enhanced Multi-Interest Learning for Multi-Behavior Sequential Recommendation**, submitted to *ESWA*.

## Requirements
The code is built on Pytorch and the [RecBole](https://github.com/RUCAIBox/RecBole) benchmark library. Run the following code to satisfy the requeiremnts by pip:

`pip install -r requirements.txt`


## Datasets
##### Download the three public datasets we use in the paper at:
Tmll:https://tianchi.aliyun.com/dataset/140281
IJCAI:https://tianchi.aliyun.com/dataset/42
UserB:https://tianchi.aliyun.com/dataset/649


##### Unzip the datasets and move them to *./dataset/*

## Run MBHT

`python run_HEML.py --model=[HEML] --dataset=[tmall_beh] --gpu_id=[0] --batch_size=[2048]`, where [value] means the default value.

## Tips
- Note that we modified the evaluation sampling setting in `recbole/sampler/sampler.py` to make it static.
- Feel free to explore other baseline models provided by the RecBole library and directly run them to compare the performances.
