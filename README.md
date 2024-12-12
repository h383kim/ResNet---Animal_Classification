# ResNet (2015)

***“Deep Residual Learning for Image Recognition” - 2015***

Link to Paper:

[https://arxiv.org/pdf/1409.1556v6](https://arxiv.org/pdf/1512.03385v1)


# 1. Introduction

> *Contributions*
> 

---

- Ensemble model won the 1st Place in classification task in ImageNet Large Scale Visual Recognition Challenge(LSVRC) in 2015 with **3.57%** top-5 error.
- Also won ILSVRC & COCO 2015 competitions, where the model won the 1st places on the tasks of *ImageNet detection*, *ImageNet localization*, *COCO detection*, and *COCO segmentation*
- Can go VERY deep. 8 times more deeper than VGG19. (i.e. 152 layers)
- Introduces *deep residual learning* framework to address the degradation problem.

<br><br><br>

> Motivation
> 

---

<p align="center">
    <img src=https://github.com/h383kim/ResNet/blob/main/images/image1.png>
</p>

- **Gradients Vanishing / Explosion is not the problem anymore:**


>> *“Vanishing/Exploding gradients problem has been largely addressed by normalized initialization and intermediate normalization layers, which enable networks with tens of layers to start converging for stochastic gradient descent (SGD) with backpropagation”*



>> *“We argue that this optimization difficulty is unlikely to be caused by vanishing gradients. These plain networks are trained with BN, which ensures forward propagated signals to have non-zero variances. We also verify that the backward propagated gradients exhibit healthy norms with BN. So neither forward nor backward signals vanish.”*
>

- **It is not even the problem of *overfitting*:**


>>“With the network depth increasing, accuracy gets saturated and then degrades rapidly.”

If it was *overfitting,* training error should not increase (as overfitting is the over training on the train dataset, training accuracy should increase)

<br><br><br>

>**Degradation Problem**

---

<p align="center">
    <img src=https://github.com/h383kim/ResNet/blob/main/images/image2.png>
</p>

When training very deep neural networks, it has been observed that adding more layers can sometimes lead to higher training error, not necessarily better performance.

So it brings up a new type of problem to deal with, ***degradation problem.***

**Conceptual Construction vs. Reality**
> 
- **Theoretical Observation:**
    - Suppose we have a shallower neural network that we can train effectively.
    - If we create a deeper neural network by adding more layers to this shallower network, theoretically, we can construct a solution for the deeper network.
    - This constructed solution involves setting the **added layers to identity mappings** (i.e., they do nothing to the inputs), and **copying the weights** from the shallower network to the corresponding layers in the deeper network.
    - This suggests that, theoretically, the deeper network should not have a higher training error than the shallower one because it can at least replicate the shallower network's behavior and pass it through using identity mappings.
- **Experimental Finding:**
    - In practice, however, when we train these deeper networks using current solvers (gradient descent, architectures, … etc.), we find that they do not perform as well as expected.

Theoretically, deeper network has the potential to perform at least as well as the shallower network because, in the worst-case scenario, it can behave exactly like the shallower network by using identity mappings and copied weights. In practice, however, optimization methods (like stochastic gradient descent) often fail to find such solutions in deeper networks.


<p align="center">
    <img src=https://github.com/h383kim/ResNet/blob/main/images/image3.png>
</p>

Residual Learning is introduced to address this practical challenge.

<br>

**Hypothesis**

Let the stacked nonlinear layers fit another mapping of $F(x) := H(x)−x$. The original mapping is recast into $F(x)+x$.

![**(Residual Block | Shortcut Connections | Skip-Connection)**](https://github.com/h383kim/ResNet/blob/main/images/image4.png)

**(Residual Block | Shortcut Connections | Skip-Connection)**

> **Claim:** We hypothesize that it is easier to optimize the residual mapping than to optimize the original, unreferenced mapping.
> 

---

**WHY?** 

To the extreme, if an identity mapping were optimal, it would be easier to push
the residual to zero than to fit an identity mapping by a stack of nonlinear layers.

**Learning $F(x)=0 \text{ vs. } H(x)=x$**

**Residual Network (Learning $F(x)=0$):** 

The network needs to learn that the residual $F(x)$ is zero. In practice, this means the network just needs to push the residuals to zero, which is a simpler task because:

- It starts with an initial guess of around zero (Weight initialization distribution around 0), which might already be close to the optimal solution.
- The adjustments needed to achieve zero residuals are typically small and localized, making the optimization process smoother and more stable.

**Standard(Plain) Network (Learning $H(x)=x$):** 

The network needs to learn the exact identity mapping from scratch, which involves:

- Learning a complex mapping through potentially many layers of nonlinear transformations.
- This can be much more challenging because the network needs to figure out how to effectively propagate the identity through each nonlinear layer, which can lead to difficulties like vanishing gradients and longer convergence times.

<br><br><br>

# 2. Architecture
<br><br>
<p align="center">
    <img src=https://github.com/h383kim/ResNet/blob/main/images/image10.png>
</p>

<br><br>
## **Residual Block**


![**(Residual Block | Shortcut Connections | Skip-Connection)**](https://github.com/h383kim/ResNet/blob/main/images/image4.png)

**(Residual Block | Shortcut Connections | Skip-Connection)**

---

**The shortcut connection is done by element-wise addition which is negligible computational cost**

> **The paper says:**
> 

---

**1)** Our extremely deep residual nets are easy to optimize, but the counterpart “plain” nets (that simply stack layers) exhibit higher training error when the depth increases; 

- This seems to be achieved by having the model to learn and granularly control which neurons will just pass through identity mappings as it became a lot easier by the skip-connection (Again, learning $F(x)=0$ is easy)

**2)** Our deep residual nets can easily enjoy accuracy gains from greatly increased depth, producing results substantially better than previous networks.

- While preventing the degrading problem, it still achieves and take advantages of better accuracy of having deeper and deeper layers.

<br><br>

## **Projection Shortcut**
> 


Refers to the use of a linear transformation (typically via a $1 \times 1$ convolutional layer) to match the dimensions of the input and the output feature maps within a residual block. This allows the input to be added to the output of the residual block even when their dimensions differ, such as when the number of channels changes or when downsampling is performed.

The projection is applied to the input $`\mathbf{x}`$ when the dimensions of $`\mathbf{x}`$ and  $\mathcal{F(\mathbf{x})}$ are not the same. This ensures that the addition in the residual block, where  $\mathcal{H}(\mathbf{x}) = \mathcal{F}(\mathbf{x}) + \mathbf{x}$ , is valid.


<p align="center">
    <img src=https://github.com/h383kim/ResNet/blob/main/images/image5.png>
</p>

The dimensions of $`\mathbf{x}`$ and $`\mathcal{F}`$ must be equal. If this is not the case (*e.g*., when changing the input/output channels), we can perform a linear projection $`W_s`$ by the shortcut connections to match the dimensions:

$$
\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + W_s \mathbf{x}
$$

We can also use a square matrix $W_s$. But, by experiments, identity mapping is sufficient for addressing the degradation problem and is economical, and thus $W_s$ is only used when matching dimensions.


**Options for Shortcuts in Residual Blocks**

1.	**Option (A): Zero-padding shortcuts for increasing dimensions**

**Zero-padding:** When the dimensions increase (e.g., when the number of channels increases or when downsampling), the **input** is **padded with zeros** to match the dimensions of the output.

(This approach is parameter-free because zero-padding does not involve learnable parameters.)

2.	**Option (B): Projection shortcuts for increasing dimensions**

**Projection shortcuts:** When the dimensions increase, a convolutional layer (projection) is used to match the input dimensions to the output dimensions. This layer has learnable parameters.

3.	**Option (C): All shortcuts are projections**

**Projection shortcuts:** For all cases, whether the dimensions increase or remain the same, a convolutional layer is used to project the input to the required output dimensions.

(This approach is NOT parameter-free, thus increases complexities and costs).

<p align="center">
    <img src=https://github.com/h383kim/ResNet/blob/main/images/image6.png>
</p>

**Small differences among A/B/C** indicate that projection shortcuts are NOT essential for addressing the degradation problem. So 50, 101, 152-layer models do not use C to reduce memory/time complexity and model sizes. Identity shortcuts are particularly important for not increasing the complexity of the bottleneck architectures that are introduced below.

<br><br>

## **Bottleneck**

<p align="center">
    <img src=https://github.com/h383kim/ResNet/blob/main/images/image7.png width="200">
</p>

For deeper models(50, 101, 152-layers) incorporate **“bottleneck blocks”** to reduce the **complexity** and **training time**. The three layers are $1×1, 3×3,$ and $1×1$ convolutions, where the $1×1$  layers are responsible for reducing and then increasing (restoring) dimensions, leaving the $3×3$ layer a bottleneck with smaller input/output dimensions

![(Left: regular, Right: bottleneck)](https://github.com/h383kim/ResNet/blob/main/images/image8.png)

(Left: regular, Right: bottleneck)

If projection shortcut is used instead of identity shortcut, time complexity/model size would be doubled. So the paper only use identity shortcut.




FLOPS is a unit of speed. FLOPs is a unit of amount.


<br><br>

## **Residual Block As an Appropriate Preconditioning**



### 1. Optimal Function

- **$H(x)$** is the ultimate output of the network block, which is the sum of the input $`x`$ and the residual function $`F(x)`$:
$H(x)=F(x)+x$
- **Optimal Function** refers to the best possible function that the network is trying to learn to minimize the training error. This optimal function is $H(x)$, which is the final output after considering both the residual $F(x)$ and the identity mapping $x$.

### 2. Understanding the Sentence

**"If the optimal function is closer to an identity mapping than to a zero mapping, it should be easier for the solver to find the perturbations with reference to an identity mapping, than to learn the function as a new one."**

Let's break it down:

- **Optimal Function Closer to Identity Mapping:** If the best function that the network should learn (i.e. $H(x)$) is very similar to the input itself ($H(x)≈x$), this means $F(x)$ (the residual) should be very small or close to zero because:

$$
H(x)=F(x)+x \text{ and if } H(x)≈x \text{ then } F(x)≈0
$$

- **Easier to Find Perturbations:** In the residual network, the task for the solver (optimizer) is to learn the small differences $F(x)$ rather than learning $H(x)$ directly. This means the network can start with an identity mapping and only needs to adjust the small deviations (perturbations) to reach the optimal function:

$$
\text{Learn } F(x)≈0 \text{ (small adjustments) instead of learning } H(x) \text{ from scratch}
$$

- **Learn the Function as a New One:** In traditional networks (without residuals), the network would need to learn $H(x)$ directly from scratch. This is harder because it involves more complex transformations through multiple layers.

### 3. Small Responses of Learned Residual Functions

**"Learned residual functions in general have small responses, suggesting that identity mappings provide reasonable preconditioning."**

Explanation:

- **Identity Mappings as Preconditioning:** Since $F(x)$ is small, the network's output $H(x)=F(x)+x$ is close to the input $x$. This indicates that starting with an identity mapping $x$ and learning small adjustments $F(x)$ is a good initial guess (preconditioning). It sets the network closer to the optimal solution from the beginning, making it easier to learn the final function.
- **Small Responses of $F(x)$:** In experiments, it has been observed that the actual values of $F(x)$ learned by the network are generally small. This means that the changes or corrections $`F(x)`$ makes to the input $`x`$ are minor.

$$F(x) \approx 0 \text{ for most inputs}$$

<br><br>

Local Responses and Residual Blocks

---

<p align="center">
    <img src=https://github.com/h383kim/ResNet/blob/main/images/image9.png>
</p>


<br><br><br>

# 3. Experiments

Datasets downloaded from Kaggle

Animals-10 Dataset consists of ~25k images of 10 classes of animals:

[https://www.kaggle.com/datasets/alessiocorrado99/animals10?resource=download](https://www.kaggle.com/datasets/alessiocorrado99/animals10?resource=download)
