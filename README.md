# AutoEdge

A PyTorch implementation of AutoEdge based on SPL paper [Zero-shot Sketch-based Image Retrieval Guided by Auto Edge]().

![Network Architecture](result/structure.png)

## Requirements

- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)

```
conda install pytorch=1.8.1 torchvision torchaudio cudatoolkit -c pytorch
```

## Dataset

[Sketchy Extended](http://sketchy.eye.gatech.edu) and 
[TU Berlin](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/) datasets are used in this repo, you could 
download these datasets from official websites, or download them from 
[MEGA](https://mega.nz/folder/IooQkZRJ#jLYcZ5PFK9jzxLN4FuOopg). The data directory structure is shown as follows:

 ```
├──sketchy
   ├── train
       ├── sketch
           ├── airplane
               ├── n02691156_58-1.jpg
               └── ...
           ...
       ├── photo
           same structure as sketch
   ├── val
      same structure as train
      ...
├──tuberlin
   same structure as sketchy
   ...
```

## Usage

```
python main.py --data_name tuberlin
optional arguments:
--data_root                   Datasets root path [default value is 'data']
--data_name                   Dataset name [default value is 'cufsf'](choices=['sketchy', 'tuberlin'])
--proj_dim                    Projected feature dim for computing loss [default value is 128]
--temperature                 Temperature used in softmax [default value is 0.1]
--batch_size                  Number of images in each mini-batch [default value is 32]
--iters                       Number of bp over the model to train [default value is 10000]
--save_root                   Result saved root path [default value is 'result']
```

## Benchmarks

The models are trained on one NVIDIA GTX TITAN (12G) GPU. `Adam` is used to optimize the model, `lr` is `1e-3`
and `weight decay` is `1e-6`. all the hyper-parameters are the default values.

<table>
<thead>
  <tr>
    <th rowspan="2">Backbone</th>
    <th rowspan="2">Dim</th>
    <th colspan="4">Sketchy Extended</th>
    <th colspan="4">TU Berlin</th>
    <th rowspan="2">Download</th>
  </tr>
  <tr>
    <td align="center">mAP@200</td>
    <td align="center">mAP@all</td>
    <td align="center">P@100</td>
    <td align="center">P@200</td>
    <td align="center">mAP@200</td>
    <td align="center">mAP@all</td>
    <td align="center">P@100</td>
    <td align="center">P@200</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">VGG16</td>
    <td align="center">512</td>
    <td align="center">29.33</td>
    <td align="center">33.33</td>
    <td align="center">45.33</td>
    <td align="center">58.67</td>
    <td align="center">29.33</td>
    <td align="center">33.33</td>
    <td align="center">45.33</td>
    <td align="center">58.67</td>
    <td align="center"><a href="https://pan.baidu.com/s/1yZhkba1EU79LwqgizDzTUA">agdw</a></td>
  </tr>
  <tr>
    <td align="center">ResNet50</td>
    <td align="center">2048</td>
    <td align="center"><b>69.33</b></td>
    <td align="center"><b>73.33</b></td>
    <td align="center"><b>81.33</b></td>
    <td align="center"><b>88.00</b></td>
    <td align="center"><b>69.33</b></td>
    <td align="center"><b>73.33</b></td>
    <td align="center"><b>81.33</b></td>
    <td align="center"><b>88.00</b></td>
    <td align="center"><a href="https://pan.baidu.com/s/139IHtS2_tOZcEK2Qgt-yQw">5dzs</a></td>
  </tr>
</tbody>
</table>

## Results

### Sketchy Extended

![synthia](result/sketchy.png)

### TU Berlin

![cityscapes](result/tuberlin.png)
