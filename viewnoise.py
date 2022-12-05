from pickle import TRUE
import sys, glob, os, imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
import helper.canon_supervised_dataset as dset
import helper.post_processing as pp
import helper.utils as ut
import matplotlib
import helper.gan_helper_fun as gh
from skimage import exposure
from PIL import Image
matplotlib.rcParams.update({'font.size': 22})

device = 'cuda:0'
gpu = 0
base_folder = 'saved_models/'
chkp_name ='noise_generator'
generator = gh.load_from_checkpoint_ab(base_folder + chkp_name, gpu).to(device)
generator.keep_track = True

#options: gan_gray, gan_color, natural 
dataset_arg = 'natural'
# Change this filepath to point to your downloaded data directory:
filepath_data = 'data/'
dataset_list_test = dset.get_dataset_noise_visualization(dataset_arg, filepath_data)

test_loader = torch.utils.data.DataLoader(dataset=dataset_list_test, 
                                           batch_size=1,
                                           shuffle=False)
for j in range(20):

    sample = dataset_list_test.__getitem__(j) #change the input image 

    with torch.no_grad():
        # clean_raw = gh.t32(sample['gt_label_nobias'].unsqueeze(0).to(device))
        # clean_raw = sample['gt_label_nobias'].unsqueeze(0).to(device)

        # real_noisy = sample['noisy_input'].unsqueeze(0).cpu().detach().numpy().transpose(0,2,3,1)

        # clean_patch = sample['gt_label_nobias'].unsqueeze(0).cpu().detach().numpy().transpose(0,2,3,1)

        # generator.indices = [0,0]
        # gen_noisy = generator(clean_raw, False)
        # gen_noisy_np = gen_noisy.cpu().detach().numpy().transpose(0,2,3,1)
        clean_raw = gh.t32(sample['gt_label_nobias'].unsqueeze(0).to(device))
        real_noisy = sample['noisy_input'].cpu().detach().numpy().transpose(1,2,3,0)
        clean_patch = sample['gt_label_nobias'].cpu().detach().numpy().transpose(1,2,3,0)

        generator.indices = [10,10]
        gen_noisy = generator(clean_raw, False)
        gen_noisy_np = gen_noisy.cpu().detach().numpy().transpose(0,2,3,1)


    ind = 0
    sz = 256
    pp = [200,200]

    to_plot = [clean_patch[ind], real_noisy[ind], gen_noisy_np[ind]]
    titles = ['clean patch', 'real noise', 'generated noise']

    fig, ax = plt.subplots(1,3, figsize = (20,10))
    for i in range(0,3):
        # ax[i].imshow(to_plot[i][pp[0]:pp[0]+sz,pp[1]:pp[1]+sz,0:3])#**(1/2.2)) #what is this number used for 
        # ax[i].axis('off')
        # ax[i].set_title(titles[i])
        noiseimage=to_plot[i][pp[0]:pp[0]+sz,pp[1]:pp[1]+sz,0:3]**(1/2.2)
        save_name='noiseimageunderstarori/'+f'testimage{j}_{i}.png'
        Image.fromarray((np.clip(noiseimage,0,1) * 255).astype(np.uint8)).save(save_name)
    # plt.show()

    # c = 0
    # all_noise_comps = list(generator.all_noise.keys())
    # fig, ax = plt.subplots(1,len(all_noise_comps), figsize = (20,10))
    # for i in range(0,len(all_noise_comps)):
    #         noise_curr = generator.all_noise[all_noise_comps[i]][0].transpose(1,2,0)[...,0:3]
    #         ax[i].imshow(noise_curr[...,c])
    #         ax[i].set_title(all_noise_comps[i])
    #         ax[i].axis('off')
    # plt.show()