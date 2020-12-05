# Kindling the Darkness:A Practical Low-light Image Enhancer 


### 说明
- Kindling the Darkness:A Practical Low-light Image Enhancer
- pytorch 复现
- 源代码地址：[https://github.com/zhangyhuaee/KinD](https://github.com/zhangyhuaee/KinD).


### 环境
- Linux
- Python 3.6
- pytorch 1.3.1


### 测试
- 测试文件为 `./test_decom2` 
- 测试结果保存在: `./final_result`.
```bash
python test_run.py
```


### 模型
- 模型文件`./model3.py` 
以下是每个网络对应的模型
```bash
pt_decomposition.py -> MyNet_2000_re_best.pkl
pt_restore.py       -> MyNet_retore1000_best.pkl
pt_adjust.py        -> MyNet_adjust_new_2000_best.pkl
```
### 训练
算法总体框架由三个网络结构组成：pt_decomposition.py 、pt_restore.py 和 pt_adjust.py
三个网络结构分开训练
```bash
python pt_decomposition.py （必须先训练）
python pt_restore.py  （训练顺序不分先后）
python pt_adjust.py   （训练顺序不分先后）
```


