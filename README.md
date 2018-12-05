# Pytorch Converter
Pytorch model to [ncnn](https://github.com/Tencent/ncnn)

## Usage
- Put your own Pytorch network .py file and mode param file in `ModelFiles/`
- Change the path in `code/run.py`
- Put you pytorch test code at `test_pytorch/` and complete test ncnn test code at `test_ncnn/` to confirm your ncnn mode correct

## Warnings
  - **Mind the difference on ceil_mode of pooling layer among Pytorch and Caffe, ncnn**
    - You can convert Pytorch models with all pooling layer's ceil_mode=True.
    - Or compile a custom version of Caffe/ncnn with floor() replaced by ceil() in pooling layer inference.

  - **Python Errors: Use Pytorch 0.2.0 Only to Convert Your Model**
    - Higher version of pytorch 0.3.0, 0.3.1, 0.4.0 seemingly have blocked third party model conversion.
    - Please note that you can still TRAIN your model on pytorch 0.3.0~0.4.0. The converter running on 0.2.0 could still load higher version models correctly.

