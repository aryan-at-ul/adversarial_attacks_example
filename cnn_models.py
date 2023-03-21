# -*- coding: utf-8 -*-
import os
try:
    import numpy as np
    import torch
    import torchvision
    assert int(torch.__version__.split(".")[1]) >= 12, "torch version should be 1.12+"
    assert int(torchvision.__version__.split(".")[1]) >= 13, "torchvision version should be 0.13+"
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
except:
    print(f"[INFO] torch/torchvision versions not as required, installing nightly versions.")
    os.system("pip install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113")
    import torch
    import torchvision
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")

# Continue with regular imports
import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms

# Try to get torchinfo, install it if it doesn't work
try:
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it.")
    os.system("pip install -q torchinfo")
    from torchinfo import summary

# Try to import the going_modular directory, download it from GitHub if it doesn't work
try:
    from going_modular.going_modular import data_setup, engine
except:
    # Get the going_modular scripts
    print("[INFO] Couldn't find going_modular scripts... downloading them from GitHub.")
    os.system("git clone https://github.com/mrdbourke/pytorch-deep-learning")
    os.system("mv pytorch-deep-learning/going_modular .")
    os.system("rm -rf pytorch-deep-learning")
    from going_modular.going_modular import data_setup, engine


# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device



import os
import zipfile

from pathlib import Path

import requests

# Setup path to data folder
data_path = Path("data/")
image_path = "/home/melkor/projects/adversarial_attacks_example/chest_xray_fgsm" 



# Setup Dirs
train_dir = image_path + "/train"
test_dir = image_path + "/test"


manual_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # 1. Reshape all images to 224x224 (though some models may require different sizes)
    transforms.ToTensor(), # 2. Turn image values to between 0 & 1 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                         std=[0.229, 0.224, 0.225]) # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
])


# Create training and testing DataLoaders as well as get a list of class names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=manual_transforms, # resize, convert images to between 0 & 1 and normalize them
                                                                               batch_size=32) # set mini-batch size to 32

train_dataloader, test_dataloader, class_names


#DenseNet121_Weights
# Get a set of pretrained model weights
weights = torchvision.models.DenseNet161_Weights.DEFAULT # .DEFAULT = best available weights from pretraining on ImageNet
weights


# Get the transforms used to create our pretrained weights
auto_transforms = weights.transforms()
auto_transforms


# Create training and testing DataLoaders as well as get a list of class names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=auto_transforms, # perform same data transforms on our own data as the pretrained model
                                                                               batch_size=32) # set mini-batch size to 32

train_dataloader, test_dataloader, class_names


# weights = torchvision.models.DenseNet121_Weights.DEFAULT # .DEFAULT = best available weights 
# weights = torchvision.models.DenseNet161_Weights.DEFAULT # .IMAGENET = best available weights from pretraining on ImageNet
# model = torchvision.models.densenet161(weights=weights).to(device)

weights = torchvision.models.ResNet18_Weights.DEFAULT # .DEFAULT = best available weights
# model = torchvision.models.resnet18(weights=weights).to(device)
model = torchvision.models.resnet18(pretrained=True).to(device)




# for denset models
# for param in model.features.parameters():
#     param.requires_grad = True #this was false in the original

for param in model.parameters():
    param.requires_grad = True #this was false in the original  

# Print a summary using torchinfo (uncomment for actual output)
summary(model=model, 
        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)
print("this summary with parmans true")


# Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
# for param in model.features.parameters():
#     param.requires_grad = True #this was false in the original

for param in model.parameters():
    param.requires_grad = True #this was false in the original  


# Set the manual seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Get the length of class_names (one output unit for each class)
output_shape = len(class_names)
print(output_shape)
# Recreate the classifier layer and seed it to the target device
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=512, 
                    out_features=output_shape, # same number of output units as our number of classes
                    bias=True)).to(device)


# summary(model, 
#         input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
#         verbose=0,
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]
# )


# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Set the random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# Setup training and save the results
results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=10,
                       device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

torch.save(model, 'model_resnetadv_head.pt')


# # Get the plot_loss_curves() function from helper_functions.py, download the file if we don't have it
# try:
#     from helper_functions import plot_loss_curves
# except:
#     print("[INFO] Couldn't find helper_functions.py, downloading...")
#     with open("helper_functions.py", "wb") as f:
#         import requests
#         request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
#         f.write(request.content)
#     from helper_functions import plot_loss_curves

# # Plot the loss curves of our model
# plot_loss_curves(results)



# from typing import List, Tuple

# from PIL import Image

# # 1. Take in a trained model, class names, image path, image size, a transform and target device
# def pred_and_plot_image(model: torch.nn.Module,
#                         image_path: str, 
#                         class_names: List[str],
#                         image_size: Tuple[int, int] = (224, 224),
#                         transform: torchvision.transforms = None,
#                         device: torch.device=device):
    
    
#     # 2. Open image
#     img = Image.open(image_path)
#     if len(img.size) < 3:
#         img = np.stack((img,)*3, axis=-1)
#     # 3. Create transformation for image (if one doesn't exist)
#     if transform is not None:
#         image_transform = transform
#     else:
#         image_transform = transforms.Compose([
#             transforms.Resize(image_size),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225]),
#         ])

#     ### Predict on image ### 

    # 4. Make sure the model is on the target device
#     model.to(device)

#     # 5. Turn on model evaluation mode and inference mode
#     model.eval()
#     with torch.inference_mode():
#       # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
#       transformed_image = image_transform(img).unsqueeze(dim=0)

#       # 7. Make a prediction on image with an extra dimension and send it to the target device
#       target_image_pred = model(transformed_image.to(device))

#     # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
#     target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

#     # 9. Convert prediction probabilities -> prediction labels
#     target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

#     # 10. Plot image with predicted label and probability 
#     plt.figure()
#     plt.imshow(img)
#     plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
#     plt.axis(False);



# # Get a random list of image paths from test set
# import random
# num_images_to_plot = 3
# test_image_path_list = list(Path(test_dir).glob("*/*.jpeg")) # get list all image paths from test data 
# test_image_path_sample = random.sample(population=test_image_path_list, # go through all of the test image paths
#                                        k=num_images_to_plot) # randomly select 'k' image paths to pred and plot

# # Make predictions on and plot the images
# for image_path in test_image_path_sample:
#     pred_and_plot_image(model=model, 
#                         image_path=image_path,
#                         class_names=class_names,
#                         # transform=weights.transforms(), # optionally pass in a specified transform from our pretrained model weights
#                         image_size=(224, 224))
