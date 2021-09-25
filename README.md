# Kindling the Darkness:A Practical Low-light Image Enhancer 



### pytorch reimplementation
### Official code ：[https://github.com/zhangyhuaee/KinD](https://github.com/zhangyhuaee/KinD).


### Enviroment
- Linux
- Python 3.6
- pytorch 1.3.1


### Test
- Test image path:`./test_decom2` 
- The result of path: `./final_result`.
```bash
python test_run.py
```


### Model
- The model file:`./model3.py` 
- Pretrained models are as follows:
```bash
pt_decomposition.py -> MyNet_2000_re_best.pkl
pt_restore.py       -> MyNet_retore1000_best.pkl
pt_adjust.py        -> MyNet_adjust_new_2000_best.pkl
```
### Train
- Model files are as follows：pt_decomposition.py 、pt_restore.py and pt_adjust.py
- Training separately
```bash
python pt_decomposition.py （training first）
python pt_restore.py  
python pt_adjust.py   
```
### The results of Decom and Adjust



