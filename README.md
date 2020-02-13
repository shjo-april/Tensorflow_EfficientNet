# EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

# # Abstract

# # Commands
```
$ python Generate_TFRecord.py
```

```
## single gpu
$ python Train.py --use_gpu 0

## multiple gpus
$ python Train.py --use_gpu 0,1,2,3,4,5,6,7
```

# # Hardware and Software
- Tensorflow 1.13.1
- OpenCV

- Single GPU : GTX 1080 Ti
- Multiple GPUs : GTX 1080 Ti x 8

# # References
- Official Tensorflow Implementation [tensorflow/tpu/Efficientnet] [[Code]](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)