# MSc-RetinaNet-project
 Part of the codes for my M.Sc. project "*RetinaNet for Lesion Detection in Medical Images*", the model that I used can be found in this [repo](https://github.com/fizyr/keras-retinanet) (keras implementation).
## Aim and objectives
- Apply RetinaNet to [INbreast](https://www.sciencedirect.com/science/article/abs/pii/S107663321100451X) dataset to perform lesion detection task (focus on mass), check its applicability and robustness.
- Make modifications to the default setting of some parameters of the network, compare and analyze corresponding results.
- Speed up the training process via adjusting or proposing...
  - [focal loss](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html) parameters
  - anchor configurations
  - backbone networks
  - splitting ratio in the training dataset
  - training strategy...

##  RetinaNet architecture

<p align = "center">
 <img src = "RetinaNet_porject_figs/RetinaNet.jpg" width = "700"/>
</p>
<p align="center">
 <sub>Fig. 1: Architecture of the RetinaNet.</sub>
</p>

## Focal loss
For the binary-classification problems, traditionally we apply cross Entropy (CE) as the loss function, which is defined as follows:

$$\begin{equation} 
  CE(p, y) = \begin{cases} 
      -\log_2 p, & \text{if } y = 1 \\
      -\log_2 (1-p), & \text{otherwise}
   \end{cases}
\end{equation}$$


where $y$ is the true label for the samples, which takes on value $1$ or $-1$. $p$ is the probability of the sample to be from class $1$ estimated by the model, which has a value in the range $[0,1]$. If we redefine another variable $P_t$, as follows:

$$\begin{equation}
  P_t = \begin{cases} 
       p, & \text{if } y = 1 \\
      1-p, & \text{otherwise}
   \end{cases}
\end{equation}$$

then the CE loss can be rewriten as: $CE(p,y) = - \log_2 P_t$. For small object detection tasks, this has two major issues:
- Large number of background samples, or in other words, easy-classified samples ($P_t \geq 0.5$), contribute too much loss.
- Because the total loss is almost "dominated" and "overwhelmed" by the loss contributed by the background samples, model can hardly learn from the foreground samples.

Such a problem can be summarized as "imbalanced class issue". 
In order to address this problem, in focal loss, a weighting factor is multiplied: $FL(P_t) = -\alpha_t(1-P_t)^\gamma \log_2 P_t$, where 

$$\begin{equation} 
  \alpha_t = \begin{cases} 
       \alpha, & \text{if } y = 1 \\
      1-\alpha, & \text{otherwise}
   \end{cases}
\end{equation}$$

> In the focal loss, when a sample is misclassified, which implies $P_t$ is small, the value of the weighting factor is close to $1$, making the FL similar to CE. When an easy sample is correctly classified, the weighting factor tends to $0$, which will scale down the loss contributed by the easy-classified samples, making the model focus more on and learn more from the hard-classified samples. Compared with CE, in FL, the loss contributed by the misclassified samples is almost unchanged while the loss contributed by the easy-classified samples was largely scaled down. The smooth factor $\gamma$ is here for adjusting the extent to which we want our down-scaling modulator to work.


## Some pre-processing
- Duplicate the one-channel gray-scale mammogram into three channels to mimic a colored image, and feed into the model.
- Normalize the input mammogram as
  $$M_{out} = \frac{M_{in}-\mu}{\sigma}$$
  where $\mu$ and $\sigma$ are the `mean` and `standard deviation` of the mammograms' pixel value in the training dataset.
> Note: When calculating the values of $\mu$ and $\sigma$, the pixels of the black background on the mammogram, which have values of `0`, are ignored. This is due to the reason that on a mammogram, the area that deserves our attention is the body tissue part, not the background. At the same time, most of the time, the black background would take up more than half the space of a mammogram, and the calculated statistical representation of the mammogram will be biased if the background information is counted.

## Two-stage training strategy
Due to the large size of the mammogram, in the training stage, for each epoch, the GPU memory is only able to hold one single training mammogram, this makes it difficult to stabilize the trained mean and trained variance in the batch normalization layer.

- Method 1: Resize the input mammogram:
  - Fail to exploit the high resolution provided by medical images in the INbreast dataset.
  - Comparably bad performance.
- Method 2: Two-phase training strategy:
  - Phase 1: Pre-train the model on image patches (cropped mammogram).
  - Phase 2: Fine-tune the model on full-size mammograms (freeze the backbone network).

## Image patches (used for pre-training)
Cropped each positive sample (mammogram containing mass) five times, and each negative sample two times. When the positive sample is cropped, it is guaranteed that the cropped patch will contain the complete mass, not only a portion of the mass. All the resulting image patches are of size `1024*1024` $\text{pixel}^2$.
<p align = "center">
 <img src = "RetinaNet_porject_figs/patch.png" width = "700"/>
</p>
<p align="center">
 <sub>Fig. 2: One mammogram and one corresponding patch cropped from it.</sub>
</p>

## Some training history results
### Tuning hyperparameters of focal loss
<p align = "center">
 <img src = "RetinaNet_porject_figs/different_gamma_loss.jpg" width = "380"/> <img src = "RetinaNet_porject_figs/different_gamma_mAP.jpg" width = "400"/>
</p>

<p align="center">
 <sub>Fig. 3: Training history with different focal loss parameters settings.</sub>
</p>

| Backbone networks | Epoch number that gives best model | mAP on testing set | Training time/epoch |
|:-----------------:|:----------------------------------:|:------------------:|:-------------------:| 
| ResNet-50         | 39th epoch                         | 74.52%             | 49 mins             |
| ResNet-101        | 39th epoch                         | 68.23%             | 50 mins             |
| VGG-16            | 40th epoch                         | 44.68%             | 50 mins             |
| VGG-19            | 36th epoch                         | 58.63%             | 51 mins             |
 
<p align="center">
<sub>Table. 1: Testing mAP of models with four different backbone networks.</sub>
</p>


### Anchor optimization and two-stage training strategy
Also use the anchor optimization introduced in this [repo](https://github.com/martinzlocha/anchor-optimization).

<p align = "center">
 <img src = "RetinaNet_porject_figs/comparison_anchor_optimization.jpg" width = "400"/> <img src = "RetinaNet_porject_figs/comparison_patch_training_and_resized.jpg" width = "400"/>
</p>

<p align="center">
 <sub>Fig. 4: Training history with anchor optimization (left) and two-stage training strategy (right).</sub>
</p>

| Training strategy    | Number of trainable parameters | Training time/epoch | 
|:--------------------:|:------------------------------:|:-------------------:|
| Resize mammograms    | 36,276,717                     | 49 mins             |
| Patches pre-training | 36,276,717                     | 21 mins             | 
| Fine-tuning          | 12,821,805                     | 67 mins             | 

 
<p align="center">
<sub>Table. 2: Comparison between resized training and two-phase training (using ResNet-50 as backbone network.</sub>
</p>


Both anchor optimization and two-stage training strategy can produce more stable converged results.

## Some inference results visualization

<p align = "center">
 <img src = "RetinaNet_porject_figs/result_good.png" width = "700"/> 
</p>

<p align="center">
 <sub>Fig. 5: Visualization of some good inference results, where red and blue colors are used to indicate ground truth and predicted bounding boxes.</sub>
</p>


<p align = "center">
 <img src = "RetinaNet_porject_figs/result_bad.png" width = "700"/>
</p>

<p align="center">
 <sub>Fig. 6: Visualization of some bad inference results, it can be witnessed that no predictions are returned by the trained model.</sub>
</p>
 




 








