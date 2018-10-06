
## Setup
先配置好Anaconda3来安装Python3.5环境，tensorflow后端
```
source activate tensorflow
conda install -c conda-forge keras=2.0.1
conda install scikit-learn
conda install matplotlib
pip install imutils
pip install opencv-python
```
## Train
```
python train.py --dataset_train train_data --model model.model
```

## Predict
```
python predict.py --model model.model -dtest test_data

```
