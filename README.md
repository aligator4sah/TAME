

# Files Directory
    TAME
    |
    |--code
    |
    |--file                                 * The preprocessing codes will write some files here.
    |
    |--data                                 * Put the downloaded datasets here.
    |    |
    |    |--MIMIC
    |         |
    |         |--train_groundtruth
    |         |
    |         |--train_with_missing
    | 
    | 
    |--result                             * The imputation results and clustering results are here.
         |
         |
         |--MIMIC

# Environment
Ubuntu16.04, python2.7

Install [pytorch 1.3.0](https://pytorch.org/)



# Active Sensing
```
cd code/TAME
python inference.py

```
# inference.py 可以调用的函数
