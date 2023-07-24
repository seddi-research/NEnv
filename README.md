# NEnv: Neural Environment Maps for Global Illumination
Official repository of "NEnv: Neural Environment Maps for Global Illumination"

_[Carlos Rodriguez-Pardo*](https://carlosrodriguezpardo.es/), [Javier Fabre*](http://javierfabre.com/), [Elena Garces](https://www.elenagarces.es/), [Jorge Lopez-Moreno](http://www.jorg3.com/)_

Computer Graphics Forum (Proceedings of the Eurographics Symposium on Rendering), June 2023

### [Project Website](http://mslab.es/projects/NEnv)

### [Paper Link](https://carlosrodriguezpardo.es/projects/NEnv/assets/pdf/paper.pdf)

![teaser](https://carlosrodriguezpardo.es/projects/NEnv/assets/media/teaser_nenv.svg)
We introduce NEnv, an invertible and fully differentiable neural method which achieves high-quality reconstructions for environment maps and their probability distributions. NEnv is up to two orders of magnitude faster to sample from than analytical alternatives, providing fast and accurate lighting representations for global illumination using Multiple Importance Sampling. Our models can accurately represent both indoor and outdoor illumination, achieving higher generality than previous work on environment map approximations.



## Requirements
Please use pip to install the required packages.
``` pip install -r requirements.txt ```

## Usage

## Dataset
Please visit the [official website](http://mslab.es/projects/NEnv) to find the dataset of pre-trained models. 

## Coming Soon
We will release training code and a _PyPI_ package soon. Stay tuned for updates. 

## Citation

Please cite our publication if you end up using any of this code in your research.

```
@inproceedings{Rodriguez-Pardo_2023_EGSR,
author = {Rodriguez-Pardo, Carlos and Fabre, Javier and Garces, Elena and Lopez-Moreno, Jorge},
title = {NEnv: Neural Environment Maps for Global Illumination},
booktitle = {Computer Graphics Forum (Eurographics Symposium on Rendering Conference Proceedings)},
year = {2023},
publisher = {The Eurographics Association and John Wiley & Sons Ltd.},
ISSN = {1467-8659},
DOI = {10.1111/cgf.14883}
}
``` 
