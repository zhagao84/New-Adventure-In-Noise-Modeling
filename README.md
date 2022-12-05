# New Adventure in Noise Modeling
In recent years, deep neural network-based image and video denoising technolo- gies have achieved unprecedented progress. Existing techniques rely on large-scale noisy-clean image pairs for training the denoising models. However, capturing a real camera dataset is an unacceptable expensive, and laborious procedure. This Semester Project is dedicated to investigating a novel method to generate realistic noisy images.
In this work, inspired by Kristina et al., we introduced a new perspective for synthesizing realistic image noise. We propose to use a combination of physical- based and DNN-based (GAN)methods. The noise model is learned by using a physics-inspired noise generator and easy-to-obtain paired images.

## Setup:

The dependencies can be installed by using:
```
conda env create -f environment.yml
source activate starlight
```



## Dataset 
Our approach is trained and evaluated on a widely used real image de- noising dataset SIDD. SIDD is collected by five smartphone cameras.
In addition, used as a comparison, we also synthesize noise on DUS. 


## Noise generator training code
Our code to train our noise generator can be found in train_gan_noisemodel.py. This code takes in a small dataset of paired clean/noisy image bursts and learns a physics-informed noise model to represent realistic noise for a camera at a fixed gain setting. 

To run the training script, run the following command:
```
python train_gan_noisemodel.py --batch_size 1 --gpus 1 --noiselist shot_read_uniform_row1_rowt_fixed1_periodic
```

We include options to change the physics-inspired noise parameters. For example, if you only want to include read and shot noise, you can run the script with the options ```--noiselist shot_read```. In addition, we provide options for including or excluding the U-Net from the noise model, options for specifying the dataset, and changing the discriminator loss to operate in real or fourier space.

### Check noise generator performance after training
To check out noise generator performance, run viewnoise.py, and change the following line to go to your saved checkpoint:

```
chkp_path = 'checkpoint_name_here'
```

By default, checkpoints are saved in /saved_models/ 

## Sythesized noise pipeline
![Image text](https://github.com/zhagao84/New-Adventure-In-Noise-Modeling/blob/master/sampleimage/pipeline.png)

