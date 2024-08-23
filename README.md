# Crack detection and segmentation for masonry surfaces

This is a fork of the [crack_detection_CNN_masonry repoistory](https://github.com/dimitrisdais/crack_detection_CNN_masonry), which aims to share the code of their research about crack detection/segmentation on masonry surfaces. This fork improves upon the original code through the following:

* Provide a clearer, well-documented codebase which is easier to interpret and use.
* Separation of the input parameters into configuration files, making it significantly easier to run
easier to train multiple networks at the same time.
* Pipeline automation, no more separate building of the dataset! Simply run the code in either evaluation or training mode through CLI arguments.
* Remove unnecessary conventions like dataset folder naming.
* Dependency cleanup: clearly indicate all dependencies through a `requirements.txt` file and separate the plotting and evalution of gathered data into a separate module, which is optional to use.

The core functionality of the original repository is retained, meaning that when using the same parameters, the same results as the original repository are obtained.

## Installation

WIP

## Usage

WIP

# Acknowledgements - original work

The original repository was produced to share material relevant to the Journal paper **[Automatic crack classification and segmentation on masonry surfaces using convolutional neural networks and transfer learning](https://doi.org/10.1016/j.autcon.2021.103606)** by **[D. Dais](https://www.researchgate.net/profile/Dimitris-Dais)**,  **İ. E. Bal**, **E. Smyrou**, and **V. Sarhosis** published in **Automation in Construction**.  

The paper can be downloaded from the following links:
- [https://doi.org/10.1016/j.autcon.2021.103606](https://doi.org/10.1016/j.autcon.2021.103606)
- [https://www.researchgate.net/publication/349645935_Automatic_crack_classification_and_segmentation_on_masonry_surfaces_using_convolutional_neural_networks_and_transfer_learning
](https://www.researchgate.net/publication/349645935_Automatic_crack_classification_and_segmentation_on_masonry_surfaces_using_convolutional_neural_networks_and_transfer_learning)

In case you use or find interesting their work please cite the following journal publication:

**D. Dais, İ.E. Bal, E. Smyrou, V. Sarhosis, Automatic crack classification and segmentation on masonry surfaces using convolutional neural networks and transfer learning, Automation in Construction. 125 (2021), pp. 103606. https://doi.org/10.1016/j.autcon.2021.103606.**

``` 
@article{Dais2021,  
  author = {Dais, Dimitris and Bal, İhsan Engin and Smyrou, Eleni and Sarhosis, Vasilis},  
  doi = {10.1016/j.autcon.2021.103606},  
  journal = {Automation in Construction},  
  pages = {103606},  
  title = {{Automatic crack classification and segmentation on masonry surfaces using convolutional neural networks and transfer learning}},  
  url = {https://linkinghub.elsevier.com/retrieve/pii/S0926580521000571},  
  volume = {125},  
  year = {2021}  
}  
```

# References
The following codes are based on material provided by **[Adrian Rosebrock](linkedin.com/in/adrian-rosebrock-59b8732a)** shared on his blog (**https://www.pyimagesearch.com/**) and his books:

* `build_data.py`  
* `hdf5datasetgenerator_mask.py`  
* `hdf5datasetwriter_mask.py`
* `epochcheckpoint.py`
* `trainingmonitor.py`

- Adrian Rosebrock, Deep Learning for Computer Vision with Python - Practitioner Bundle, PyImageSearch, https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/, accessed on 24 February 2021  
- Adrian Rosebrock, Keras: Starting, stopping, and resuming training, PyImageSearch, https://www.pyimagesearch.com/2019/09/23/keras-starting-stopping-and-resuming-training/, accessed on 24 February 2021  
- Adrian Rosebrock, How to use Keras fit and fit_generator (a hands-on tutorial), PyImageSearch, https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/, accessed on 24 February 2021  

The Segmentation Models with pre-trained CNNs are implemented based on the work of **[Pavel Yakubovskiy](https://github.com/qubvel)** and his GitHub Repository https://github.com/qubvel/segmentation_models  

**DeepCrack** is implemented as provided by the corresponding [GitHub Repository](https://github.com/hanshenChen/crack-detection)  

**Deeplabv3** is implemented as provided by the corresponding [GitHub Repository](https://github.com/tensorflow/models/tree/master/research/deeplab)  

Unet is based on the implementation found in the link below:  
https://www.depends-on-the-definition.com/unet-keras-segmenting-images/  

The Weighted Cross-Entropy (WCE) Loss is based on the implementation found in the link below:  
https://jeune-research.tistory.com/entry/Loss-Functions-for-Image-Segmentation-Distribution-Based-Losses  

The Focal Loss is based on the implementation found in the link below:  
https://github.com/umbertogriffo/focal-loss-keras
