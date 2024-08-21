import os
import argparse

import numpy as np
import pandas as pd

import torch
from torchvision import datasets
from torchvision import transforms as pth_transforms

import vit_utils
import vision_transformer as vits


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Linear evaluation')
    parser.add_argument('--batch_size', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--data_path', default='/Users/ima029/Desktop/SCAMPI/Repository/data/NO 6407-6-5/labelled imagefolders/imagefolder_20', type=str, help='Path to evaluation dataset')
    parser.add_argument('--destination', default='', type=str, help='Destination folder for saving results')
    parser.add_argument('--img_size', default=224, type=int, help='The size of the images used for training the model')
    parser.add_argument('--img_size_pred', default=224, type=int, help='The size of the images used for prediction')
    args = parser.parse_args()
    
    # ============ preparing data ... ============
    
    transform = pth_transforms.Compose([
        pth_transforms.Resize((256, 256), interpolation=3),
        pth_transforms.CenterCrop(args.img_size_pred),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    ds = datasets.ImageFolder(args.data_path, transform=transform)
    
    data_loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    
    # ============ building network ... ============
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0, img_size=[args.img_size])
    vit_utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()
        
    # ============ extract features ... ============
    print("Extracting features...")

    features = []

    for samples, _ in data_loader:
        features.append(model(samples).detach().numpy())
    
    features = np.concatenate(features, axis=0)
    filenames = [os.path.basename(f[0]) for f in ds.imgs]
    
    # save features as json with filenames as keys
    features_df = pd.DataFrame(features)
    features_df['filename'] = filenames
    features_df.to_csv(os.path.join(args.destination, 'features.csv'), index=False)
