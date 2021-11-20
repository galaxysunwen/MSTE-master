# Effective Knowledge Graph Embeddings based on Multidirectional Sementics Relations for Polypharmacy Side Effects Prediction
This is the code of paper **Effective Knowledge Graph Embeddings based on Multidirectional Sementics Relations for Polypharmacy Side Effects Prediction.** *Junfeng Yao, Wen Sun, Zhongquan Jian, Qingqiang Wu, Xiaoli Wang.*

## Dependencies
- Python 3.6+
- [Tensorflow](https://tensorflow.google.cn/) 1.13.1+

## Results
The results of **MSTE** on **TWOSIDES** and **DrugBank** are as follows.
 
### TWOSIDES
| | ROC-AUC |  PR-AUC | AP@50 |
|:----------:|:----------:|:----------:|:----------:|
| MSTE | 97.44 | 96.73 | 98.86 |



### DrugBank
| | ROC-AUC |  PR-AUC | AP@n |
|:----------:|:----------:|:----------:|:----------:|
| MSTE | 99.59 |  99.48 | 99.37 |



## Running the code 

### Usage
You can train the MSTE models on the two datasets by running its corresponding training script as follows:
```
for TWOSIDES dataset: python MSTE.py
for DrugBank dataset: python MSTE_DB.py
```


## Citation
If you find this code useful, please consider citing our paper.


## Acknowledgement
We refer to the code of [TriVec](https://github.com/samehkamaleldin/pse-kge). Thanks for their contributions.

