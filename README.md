# FaceIDLight


## Description
A lightweight face-recognition toolbox and pipeline based on tensorflow-lite with MTCNN-Face-Detection 
and ArcFace-Face-Recognition. No need to install complete tensorflow, tflite-runtime is enough. All tools are
using CPU only. 

Pull request are welcome!


### Features 
- Online Face-Recognition
- Running completely on CPU
- Multi Faces
- ~4 FPS on a MacBookPro2015
- Tools for Face-Detection, -Verification and Identification


### ToDos
- [ ] GPU support
- [ ] Resolution-dependent model-selection
- [ ] Multithreading for multiple faces
- [ ] OpenCV Window freezes on MacOS when quitting


## Requirements
- [Python 3.8.8](https://www.python.org/)
- [TensorflowLite-Runtime 2.5.0](https://www.tensorflow.org/lite/guide/python)
- Additional Packages (included in FaceIDLight)
    - setuptools~=51.0.0
    - opencv-python~=4.5.1.48
    - numpy~=1.19.5
    - tqdm~=4.59.0
    - scikit-image~=0.18.1
    - matplotlib~=3.3.3
    - scipy~=1.6.0
    - scikit-learn~=0.24.0

## How to install tflite-runtime
### MacOS
You can easily install tflite-runtime from https://google-coral.github.io/py-repo/ with the following line:
```zsh
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
```

### Linux
Since pip3 `--extra-index-url` is not working properly on Linux systems, you need to directly point to a wheel 
selected for your system manually from google-corals website: https://google-coral.github.io/py-repo/tflite-runtime
```zsh
pip3 install <link-to-your-wheel>
``` 


### Windows
TODO


## How to install the FaceIDLight package
Simply install the package via pip:
```zsh
pip3 install git+https://github.com/martlgap/FaceIDLight
``` 


## Run Demo:
Run Python 3.8 and type in:
```python
from FaceIDLight.demo import Demonstrator
Demonstrator().run()
```
You can select your own directory for gallery-images (*.png and *.jpg images are supported) by simply add 
a keyword argument to the Demonstrator Class: `Demonstrator(gal_dir=<full-path-to-your-gallery>)`

Test the face-identification by simply holding a foto into camera. The provided sample_gallery includes images 
from: (Andrew_Caldecott, Anja_Paerson, Choi_Sung-hong, Elizabeth_Schumacher, 
Eva_Amurri, Jim_OBrien, Raul_Ibanez, Rubens_Barrichello, Takahiro_Mori)

## Acknowledgement
- Thanks to Iván de Paz Centeno for his [implementation](https://github.com/ipazc/mtcnn) 
  of [MTCNN](https://arxiv.org/abs/1604.02878) in [Tensorflow 2](https://www.tensorflow.org/). 
  The MTCNN tflite models are taken "as is" from his repository and were converted to tflite-models afterwards.
- Thanks to Kuan-Yu Huang for his [implementation](https://github.com/peteryuX/arcface-tf2) 
  of [ArcFace](https://arxiv.org/abs/1801.07698) in [Tensorflow 2](https://www.tensorflow.org/).
- We trained all provided models with the [MS1M](https://arxiv.org/abs/1607.08221) dataset.


## BibTex  
If you use our trained models and want to cite our work feel free to use this:
TODO