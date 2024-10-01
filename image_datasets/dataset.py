import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
import webdataset as wds
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
from torchvision import transforms
from transformers import CLIPImageProcessor

def image_resize(img, max_size=512):
    w, h = img.size
    if w >= h:
        new_w = max_size
        new_h = int((max_size / w) * h)
    else:
        new_h = max_size
        new_w = int((max_size / h) * w)
    return img.resize((new_w, new_h))

def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def crop_to_aspect_ratio(image, ratio="16:9"):
    width, height = image.size
    ratio_map = {
        "16:9": (16, 9),
        "4:3": (4, 3),
        "1:1": (1, 1)
    }
    target_w, target_h = ratio_map[ratio]
    target_ratio_value = target_w / target_h

    current_ratio = width / height

    if current_ratio > target_ratio_value:
        new_width = int(height * target_ratio_value)
        offset = (width - new_width) // 2
        crop_box = (offset, 0, offset + new_width, height)
    else:
        new_height = int(width / target_ratio_value)
        offset = (height - new_height) // 2
        crop_box = (0, offset, width, offset + new_height)

    cropped_img = image.crop(crop_box)
    return cropped_img


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=512, caption_type='json', random_ratio=False):
        self.images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if '.jpg' in i or '.png' in i]
        self.images.sort()
        self.img_size = img_size
        self.caption_type = caption_type
        self.random_ratio = random_ratio

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx]).convert('RGB')
            if self.random_ratio:
                ratio = random.choice(["16:9", "default", "1:1", "4:3"])
                if ratio != "default":
                    img = crop_to_aspect_ratio(img, ratio)
            img = image_resize(img, self.img_size)
            w, h = img.size
            new_w = (w // 32) * 32
            new_h = (h // 32) * 32
            img = img.resize((new_w, new_h))
            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)
            json_path = self.images[idx].split('.')[0] + '.' + self.caption_type
            if self.caption_type == "json":
                prompt = json.load(open(json_path))['caption']
            else:
                prompt = open(json_path).read()
            return img, prompt
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.images) - 1))
        

def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, size=512, t_drop_rate=0.05, i_drop_rate=0.05,
                 ti_drop_rate=0.05, image_root_path=""):
        super().__init__()

        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path

        self.data = json.load(open(json_file))  # list of dict: [{"image_file": "1.png", "text": "A dog"}]

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        image_file = item["image_file"]

        # read image
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        image = self.transform(raw_image.convert("RGB"))
        clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values

        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1

        return {
            "image": image,
            "text": text,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed
        }

    def __len__(self):
        return len(self.data)

def Init_WDS(wds_path,img_size):
    # Assume one epoch is 10000 batches

    tar_files = glob.glob(wds_path)
    dataset = wds.WebDataset(tar_files, resampled=True).decode("pil").shuffle(1000).to_tuple("jpg", "txt").with_epoch(10000)
    transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  
    ])
    
    clip_image_processor = CLIPImageProcessor()

    def preprocess(sample):
        try:
            raw_image, text = sample
            
            image = transform(raw_image.convert("RGB") if not raw_image.mode == "RGB" else raw_image)
            
            clip_image = clip_image_processor(images=raw_image if not raw_image.mode == "RGB" else raw_image, return_tensors="pt").pixel_values
            
            drop_image_embed = 0
            t_drop_rate = 0.05
            i_drop_rate = 0.05
            ti_drop_rate = 0.05
            rand_num = random.random()
            
            if rand_num < i_drop_rate:
                drop_image_embed = 1
            elif rand_num < (i_drop_rate + t_drop_rate):
                text = ""
            elif rand_num < (i_drop_rate + t_drop_rate + ti_drop_rate):
                text = ""
                drop_image_embed = 1

            return {
                "image": image,
                "text": text,
                "clip_image": clip_image,
                "drop_image_embed": drop_image_embed
            }
        
        except Exception as e:
            print(f"Error processing sample: {e}")
            return None

    dataset = dataset.map(preprocess).select(lambda x: x is not None)
    
    return dataset

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    texts = [example["text"] for example in data]
    # text_input_ids_2 = torch.cat([example["text_input_ids_2"] for example in data], dim=0)
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]

    return {
        "images": images,
        "texts": texts,
        # "text_input_ids_2": text_input_ids_2,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds
    }

def ip_dataset_loader(num_workers, train_batch_size, **args):
    dataset = MyDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, collate_fn=collate_fn)

def ip_wds_loader(num_workers, train_batch_size, **args):
    dataset = Init_WDS(**args)
    loader = DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=True)
    return loader