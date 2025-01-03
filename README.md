# Wasserstein-Generative-Adversarial-Network-for-Flower-Image-Generation
<p>The provided program implements a Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) for generating realistic flower images. The program utilizes TensorFlow and Keras to preprocess data, construct generator and critic architectures, and train the model using adversarial techniques.</p>

## Key Features
Data Preparation:
Supports dataset extraction and loading on both Google Colab and local machines.
Preprocesses images by resizing and normalizing pixel values to the range [-1, 1].
Model Architecture:
Generator: Uses transposed convolutional layers to upsample latent vectors into realistic images with tanh activation.
Critic: Fully convolutional design to evaluate real and fake images, enforcing Lipschitz continuity with Gradient Penalty.
## WGAN-GP Model:
Wasserstein Loss: Measures the Wasserstein distance between real and fake data distributions.
Gradient Penalty: Regularizes the critic’s gradient to improve convergence and enforce Lipschitz continuity.
Training Dynamics: Alternates between multiple critic updates and one generator update per training step.
Training and Evaluation:
Tracks critic and generator losses during training.
Displays generated images after each epoch for qualitative evaluation.

## Install the Package
pip install tensorflow matplotlib
<p>Ensure TensorFlow and Matplotlib are installed before running the program. For TensorFlow installation instructions, visit [TensorFlow Installation](https://www.tensorflow.org/install).</p>

### Running the Program
#### Dataset Preparation:
Extract the flower dataset from the zip file to the appropriate directory.
Structure the dataset into training and validation directories.
####  Google Colab:
Mount Google Drive and set paths to the dataset accordingly.
Use the provided script to unzip and organize the dataset.
#### On Local Machine:
Ensure the dataset path is correctly set in the script for training directories.
Use the provided script to unzip the dataset.
#### Execution:
Preprocess the dataset by normalizing pixel values to [-1, 1].
Build and compile the WGAN-GP model, including the generator and critic.
Train the model for the desired number of epochs.
Visualize generated images after each epoch to monitor progress.

## Contributor
<p>Lawrence Menegus</p>
