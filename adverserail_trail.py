import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.models import resnet50
import torch.nn as nn
import torch.optim as optim
import numpy as np
import PIL.Image
import matplotlib
# matplotlib.use('TkAgg')

from skimage.segmentation import slic

from skimage.segmentation import mark_boundaries
from skimage import segmentation
from skimage import data, segmentation
from skimage import io, color
from skimage.io import imread
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
from skimage.future import graph
import networkx as nx
from skimage.measure import regionprops
import os
import numpy as np
from numpy import sqrt
from eff_net_fet_extraction import fet_from_img,get_sailency_map_from_img,get_binary_mask,get_binary_mask2
import  errno
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import time
import traceback
import sys
from disf import DISF_Superpixels
from skimage import io, color


current_file_path = os.path.dirname(os.path.abspath(__file__))

data_dir = f"{current_file_path}/chest_xray"

def make_dirs(dir):
    try:
        os.makedirs(dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = resnet50(pretrained=True).to(device)
# model = torch.load("/home/melkor/projects/img_to_graph/model_densenet161_head.pt").to(device)
model = torch.load("/home/melkor/projects/img_to_graph/model_resnet_head.pt").to(device)
model.eval()


def read_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    val_dir = os.path.join(data_dir, 'val')
    train_normal_dir = os.path.join(train_dir, 'NORMAL')
    train_pneumonia_dir = os.path.join(train_dir, 'PNEUMONIA')
    test_normal_dir = os.path.join(test_dir, 'NORMAL')
    test_pneumonia_dir = os.path.join(test_dir, 'PNEUMONIA')
    val_normal_dir = os.path.join(val_dir, 'NORMAL')
    val_pneumonia_dir = os.path.join(val_dir, 'PNEUMONIA')
    return train_dir, test_dir, val_dir, train_normal_dir, train_pneumonia_dir, test_normal_dir, test_pneumonia_dir, val_normal_dir, val_pneumonia_dir


def read_image(path):
    images = []
    for one in os.listdir(path):
        print(one,"this should look correct")
        image = cv2.imread(os.path.join(path, one))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = img_as_float(image)
        images.append(image)
    return images

def read_one_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = PIL.Image.open(image_path)
    image = image.convert('RGB')    
    print(image.size,"what is this sizes")
    
    # if len(image.size) < 3:
    #     image = np.stack((image,)*3, axis=-1)
    # print(image.size,"what is this sizes")
    # else:
    #     image = image[:,:,:3]
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor


def get_image_paths(data_dir):
    train_dir, test_dir, val_dir, train_normal_dir, train_pneumonia_dir, test_normal_dir, test_pneumonia_dir, val_normal_dir, val_pneumonia_dir = read_data(data_dir)
    train_normal_image_paths = [os.path.join(train_normal_dir, f) for f in os.listdir(train_normal_dir)]
    train_pneumonia_image_paths = [os.path.join(train_pneumonia_dir, f) for f in os.listdir(train_pneumonia_dir)]
    test_normal_image_paths = [os.path.join(test_normal_dir, f) for f in os.listdir(test_normal_dir)]
    test_pneumonia_image_paths = [os.path.join(test_pneumonia_dir, f) for f in os.listdir(test_pneumonia_dir)]
    val_normal_image_paths = [os.path.join(val_normal_dir, f) for f in os.listdir(val_normal_dir)]
    val_pneumonia_image_paths = [os.path.join(val_pneumonia_dir, f) for f in os.listdir(val_pneumonia_dir)]
    # return [train_pneumonia_image_paths,test_pneumonia_image_paths,train_normal_image_paths,test_normal_image_paths, val_normal_image_paths, val_pneumonia_image_paths]
    #chnage above retunn for testing only, below one is correct
    return [train_normal_image_paths, train_pneumonia_image_paths, test_normal_image_paths, test_pneumonia_image_paths, val_normal_image_paths, val_pneumonia_image_paths]


def load_image(tain_normal_path,train_pneumonia_path,test_normal_path,test_pneumonia_path,val_normal_path,val_pneumonia_path):
    train_normal_image = read_image(tain_normal_path)
    train_pneumonia_image = read_image(train_pneumonia_path)
    test_normal_image = read_image(test_normal_path)
    test_pneumonia_image = read_image(test_pneumonia_path)
    val_normal_image = read_image(val_normal_path)
    val_pneumonia_image = read_image(val_pneumonia_path)
    return train_normal_image, train_pneumonia_image, test_normal_image, test_pneumonia_image, val_normal_image, val_pneumonia_image


# def slic_segment():
#     train_normal_image_paths, train_pneumonia_image_paths, test_normal_image_paths, test_pneumonia_image_paths, val_normal_image_paths, val_pneumonia_image_paths = get_image_paths(data_dir)
#     train_normal_image, train_pneumonia_image, test_normal_image, test_pneumonia_image, val_normal_image, val_pneumonia_image = load_image(train_normal_image_paths[0],train_pneumonia_image_paths[0],test_normal_image_paths[0],test_pneumonia_image_paths[0],val_normal_image_paths[0],val_pneumonia_image_paths[0])
#     segments = slic(train_normal_image, n_segments=100, sigma=5)
#     plt.imshow(mark_boundaries(train_normal_image, segments))
#     plt.show()


def fgsm_attack(image, epsilon, gradient):
    sign_gradient = gradient.sign()
    perturbed_image = image + epsilon * sign_gradient
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image





def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = PIL.Image.open(image_path)
    
    if len(image.size) < 3:
        image = np.stack((image,)*3, axis=-1)
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

# image_path = '/home/melkor/projects/img_to_graph/chest_xray/val/PNEUMONIA/person1946_bacteria_4874.jpeg'
# image = preprocess_image(image_path).to(device)


#revert the normalization
def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # for t, m, s in zip(tensor, mean, std):
    #     t.mul_(s).add_(m)
    return tensor


def make_adversarial_example(image, epsilon, dirs_names,class_name,data_typ,image_name):

    epsilon = 0.03  # Choose a value for epsilon (e.g., 0.03)
    image.requires_grad = True

    output = model(image)
    init_pred = output.max(1, keepdim=True)[1]

    loss = nn.CrossEntropyLoss()
    target_label = init_pred.item()
    target = torch.tensor([target_label], dtype=torch.long, device=device)
    loss_cal = loss(output, target)

    model.zero_grad()
    loss_cal.backward()

    gradient = image.grad.data
    perturbed_image = fgsm_attack(image, epsilon, gradient)


    perturbed_image = perturbed_image.squeeze(0)
    perturbed_image = unnormalize(perturbed_image.clone().detach().cpu())
    # save_image(perturbed_image, 'perturbed_image.jpg')
    return perturbed_image





def run_for_one_folder(folder_path):
    # print("check this here ", folder_path[:10])
    for one in folder_path:
        one = one.replace('._','')
        dirs_names = one.split("/")
        class_name = dirs_names[-2]
        data_typ = dirs_names[-3]
        image_name = dirs_names[-1]
        ss = time.time()
        make_dirs(f"{current_file_path}/chest_xray_fgsm/{data_typ}/{class_name}")
        # print(os.listdir(f"{current_file_path}/chest_xray_graphs/{data_typ}/{class_name}"))
        # print(image_name.split('.')[0] + ".gpickle")
        if len(os.listdir(f"{current_file_path}/chest_xray_fgsm/{data_typ}/{class_name}")) == len(os.listdir(f"{current_file_path}/chest_xray/{data_typ}/{class_name}")):
            print("all done")
            break

        if image_name.split('.')[-2]  in os.listdir(f"{current_file_path}/chest_xray_fgsm/{data_typ}/{class_name}"):
            continue
        print("="*100)
        print("done images", len(os.listdir(f"{current_file_path}/chest_xray_fgsm/{data_typ}/{class_name}")))
        print("total images present ",len(os.listdir(f"{current_file_path}/chest_xray/{data_typ}/{class_name}")))
        print(f"starting for file: {image_name}")
        # print(f"processing {class_name} {data_typ} {image_name} ---> {one}")
        image = read_one_image(one)
        epsilon = 0.03  # Choose a value for epsilon (e.g., 0.03)
        perturbed_image = make_adversarial_example(image, epsilon, dirs_names,class_name,data_typ,image_name)
        # make_dirs(f"{current_file_path}/chest_xray_graphs/{data_typ}/{class_name}")
        e = time.time()
        print(f"time for adding noise to image : {e-ss}s")
        print("="*100)
        # nx.write_gpickle(G, f"{current_file_path}/chest_xray_fgsm/{data_typ}/{class_name}" + "/" + image_name)
        save_image(perturbed_image, f"{current_file_path}/chest_xray_fgsm/{data_typ}/{class_name}" + "/" + image_name)
        # break
    return 1



def test_run():
    print("starting")
    make_dirs(current_file_path+'/'+'chest_xray_graphs')
    all_images_path = get_image_paths(data_dir)
    tasks = []
    # for one_class in all_images_path:
    #     tasks.append(str(run_for_one_folder(one_class)))
    #     # break
    # print("done")
    # with ThreadPoolExecutor(max_workers=10) as executor:
    #     future = executor.submit([run_for_one_folder(x) for x in all_images_path])
    # for result in future:
    #     print(result.result())

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    # Start the load operations and mark each future with its URL
        futures = [executor.submit(run_for_one_folder, x)  for x in all_images_path]
        for future in concurrent.futures.as_completed(futures):

            #try:
            print(future.result())
                # print("this result here",img_fet,i)
            #except Exception as exc:
            #    print(f'exception {exc} exception at folder level!!!')
            #else:
            #    print('all done for one folder')
            #end = time.time()
            #print(f"time take per folder = {end - start}")



if __name__ == '__main__':
    test_run()
