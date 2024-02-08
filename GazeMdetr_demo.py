# -*- coding: utf-8 -*-  
# Default encoding for Python3 code is utf-8. Here it is mentioned to also support Python2.x in the same time.
"""MDETR_demo.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11xz5IhwqAqHj9-XAIP17yVIuJsLqeYYJ

# MDETR - Modulated Detection for End-to-End Multi-Modal Understanding

Welcome to the demo notebook for MDETR. We'll show-case detection, segmentation and question answering

## Preliminaries

This section contains the initial boilerplate. Run it first.
"""
import sys
import argparse
import os
import xml.etree.ElementTree as ET
import torch
from PIL import Image
import numpy as np
import requests
import torchvision.transforms as T
import matplotlib.pyplot as plt
from collections import defaultdict
import torch.nn.functional as F
import numpy as np
from skimage.measure import find_contours

from matplotlib import patches,  lines
from matplotlib.patches import Polygon

parser = argparse.ArgumentParser(description='Caption format selection')
parser.add_argument('-cc', '--caption_category', type=str, choices=['A', 'B', 'C', 'D', 'E'], default='A', help='Specify a value (A, B, C, D, E) to determine the caption category. A:The, B:This is a, C:Look at the, D:Point at the, E:Pass the')
parser.add_argument('-cd', '--caption_details', type=int, choices=[1, 2, 3, 4], default=1, help='Specify a detail level as (1, 2, 3, 4) to determine the caption details. 1:pose+color+name+placement, 2:pose+name+placement, 3:color+name, 4:name')
args=parser.parse_args()

torch.set_grad_enabled(False);

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Normalize and reszie norm_map, valuesbetween (0.2,1)
transform_normMap = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize((-0.25), (1.25))
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def plot_results(pil_img, scores, boxes, labels, save_fig_path, masks=None):
    plt.figure(figsize=(16,10))
    np_image = np.array(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if masks is None:
      masks = [None for _ in range(len(scores))]
    assert len(scores) == len(boxes) == len(labels) == len(masks)
    for s, (xmin, ymin, xmax, ymax), l, mask, c in zip(scores, boxes.tolist(), labels, masks, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{l}: {s:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='white', alpha=0.8))

        if mask is None:
          continue
        np_image = apply_mask(np_image, mask, c)

        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
          # Subtract the padding and flip (y, x) to (x, y)
          verts = np.fliplr(verts) - 1
          p = Polygon(verts, facecolor="none", edgecolor=c)
          ax.add_patch(p)


    plt.imshow(np_image)
    plt.axis('off')
    if save_fig_path is not None:
        save_fig_dir = os.path.dirname(save_fig_path)
        if not os.path.exists(save_fig_dir):
            os.makedirs(save_fig_dir)
        plt.savefig(save_fig_path, bbox_inches='tight', pad_inches=0.1)
        plt.show()
    else:
        plt.show()


def add_res(results, ax, color='green'):
    #for tt in results.values():
    if True:
        bboxes = results['boxes']
        labels = results['labels']
        scores = results['scores']
        #keep = scores >= 0.0
        #bboxes = bboxes[keep].tolist()
        #labels = labels[keep].tolist()
        #scores = scores[keep].tolist()
    #print(torchvision.ops.box_iou(tt['boxes'].cpu().detach(), torch.as_tensor([[xmin, ymin, xmax, ymax]])))

    colors = ['purple', 'yellow', 'red', 'green', 'orange', 'pink']

    for i, (b, ll, ss) in enumerate(zip(bboxes, labels, scores)):
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, color=colors[i], linewidth=3))
        cls_name = ll if isinstance(ll,str) else CLASSES[ll]
        text = f'{cls_name}: {ss:.2f}'
        print(text)
        ax.text(b[0], b[1], text, fontsize=15, bbox=dict(facecolor='white', alpha=0.8))

"""## Detection

In this section, we show the performance of our pre-trained model on modulated detection.
Keep in mind that this model wasn't fine-tuned for any specific task.

We load the 'mdetr_efficientnetB5' model from torch hub

To see list of all the models you can try: torch.hub.list('ashkamath/mdetr:main')
To get info on the model you can try: torch.hub.help('ashkamath/mdetr:main', 'mdetr_efficientnetB5')
"""

model, postprocessor = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB5', pretrained=True, return_postprocessor=True)
model = model.cuda()
model.eval();

"""Next, we retrieve an image on which we wish to test the model. Here, we use an image from the validation set of COCO"""


def plot_inference(im, caption, gaze,save_fig_path):
  # mean-std normalize the input image (batch-size: 1)
  img = transform(im).unsqueeze(0).cuda()

  # propagate through the model
  memory_cache = model(img, [caption], gaze, encode_and_save=True)
  outputs = model(img, [caption], gaze, encode_and_save=False, memory_cache=memory_cache)

  # keep only predictions with 0.7+ confidence
  probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
  #keep = (probas > 0.7).cpu()
  
  # Keep the prediction with max confidence
  max_val, max_index = torch.max(probas, dim=0)
  keep = torch.zeros_like(probas, dtype=torch.bool).cpu()
  keep[max_index] = True

  # convert boxes from [0; 1] to image scales
  bboxes_scaled = rescale_bboxes(outputs['pred_boxes'].cpu()[0, keep], im.size)

  # Extract the text spans predicted by each box
  positive_tokens = (outputs["pred_logits"].cpu()[0, keep].softmax(-1) > 0.1).nonzero().tolist()
  predicted_spans = defaultdict(str)
  for tok in positive_tokens:
    item, pos = tok
    if pos < 255:
        span = memory_cache["tokenized"].token_to_chars(0, pos)
        predicted_spans [item] += " " + caption[span.start:span.end]

  labels = [predicted_spans [k] for k in sorted(list(predicted_spans .keys()))]
  plot_results(im, probas[keep], bboxes_scaled, labels, save_fig_path)

"""Let's first try to single out the salient objects in the image"""

# Original demo 
# url = "http://images.cocodataset.org/val2017/000000281759.jpg"
# im.show()
# plot_inference(im, "5 people each holding an umbrella")


# All the images and normmaps in the test sets - enter the caption through command line
file_path = "/home/suka/code/Data/annotated_MDETR_test_data"
folders = sorted([f for f in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, f))])
#folders = folders[2:]
for folder in folders:
    folder_path = os.path.join(file_path, folder)
    rgb_folders_path = os.path.join(folder_path, 'rgb_img')
    normMap_folders_path = os.path.join(folder_path, 'normMap')
    rgb_folders = sorted([f for f in os.listdir(rgb_folders_path) if os.path.isdir(os.path.join(rgb_folders_path, f))])
    normMap_folders = sorted([f for f in os.listdir(normMap_folders_path) if os.path.isdir(os.path.join(normMap_folders_path, f))])
    for i in range(min(len(rgb_folders),len(normMap_folders))):
        # input images
        images_path = os.path.join(rgb_folders_path,rgb_folders[i])
        images = sorted([f for f in os.listdir(images_path) if '.xml' not in f])
        # input caption
        annotation_path = os.path.join(images_path, 'annotation.xml')
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            if obj.find('name').text != 'head':
                obj_info = {
                    'name': obj.find('name').text,
                    'color': obj.find('color').text,
                    'pose': obj.find('pose').text,
                    'placement': obj.find('placement').text,
                    'bndbox': {
                        'xmin': int(obj.find('bndbox/xmin').text),
                        'ymin': int(obj.find('bndbox/ymin').text),
                        'xmax': int(obj.find('bndbox/xmax').text),
                        'ymax': int(obj.find('bndbox/ymax').text)
                    }
                }
        gt_bbox = [obj_info['bndbox']['xmin'], obj_info['bndbox']['ymin'], obj_info['bndbox']['xmax'], obj_info['bndbox']['ymax']]
        # input heatmaps        
        normMaps_path = os.path.join(normMap_folders_path, normMap_folders[i])
        normMaps = sorted([f for f in os.listdir(normMaps_path)])
        for j in range(min(len(images),len(normMaps))):
            im_path = os.path.join(images_path, images[j])
            im = Image.open(im_path)
            im.show()
            # Define caption templates
            caption_templates = {
                'A': {
                    1: "The {pose} {color} {name} {placement}.",
                    2: "The {pose} {name} {placement}.",
                    3: "The {color} {name}.",
                    4: "The {name}.",
                },
                'B': {
                    1: "This is a {pose} {color} {name} {placement}.",
                    2: "This is a {pose} {name} {placement}.",
                    3: "This is a {color} {name}.",
                    4: "This is a {name}.",
                },
                'C': {
                    1: "Look at the {pose} {color} {name} {placement}.",
                    2: "Look at the {pose} {name} {placement}.",
                    3: "Look at the {color} {name}.",
                    4: "Look at the {name}.",
                },
                'D': {
                    1: "Point at the {pose} {color} {name} {placement}.",
                    2: "Point at the {pose} {name} {placement}.",
                    3: "Point at the {color} {name}.",
                    4: "Point at the {name}.",
                },
                'E': {
                    1: "Pass the {pose} {color} {name} {placement}.",
                    2: "Pass the {pose} {name} {placement}.",
                    3: "Pass the {color} {name}.",
                    4: "Pass the {name}.",
                }
            }

            # Construct caption
            caption_category = args.caption_category
            caption_details = args.caption_details
            caption = caption_templates[caption_category][caption_details].format(**obj_info)
            print('caption: ', caption)

            caption_words = caption.split()
            save_fig_path = os.path.join('/home/suka/code/Data/GazeMDETR_captions_tests', str(caption_category), str(caption_details), folder, rgb_folders[i], "_".join(caption_words), images[j].split('.')[0])
            norm_map_path = os.path.join(normMaps_path, normMaps[j])
            norm_map = Image.open(norm_map_path)
            norm_map_gray = norm_map.convert('L')
            normalized_norm_map_tensor = transform_normMap(norm_map_gray)
            # Visualize norm_map tensor before downsampling
            normalized_norm_map_tensor_array = np.squeeze(normalized_norm_map_tensor.cpu().numpy())*255
            normalized_norm_map_tensor_image = Image.fromarray(normalized_norm_map_tensor_array.astype(np.uint8))
            # Visualize downsampled norm map
            downsampled_norm_map = torch.nn.functional.interpolate(normalized_norm_map_tensor.unsqueeze(0),size=(25,34), mode='bilinear', align_corners=False).squeeze(0)
            downsampled_norm_map_array = np.squeeze(downsampled_norm_map.cpu().numpy())*255
            downsampled_norm_map_image = Image.fromarray(downsampled_norm_map_array.astype(np.uint8))
            save_normMap_path = os.path.join('/home/suka/code/Data/GazeMDETR_captions_tests', str(caption_category), str(caption_details), folder, rgb_folders[i], "_".join(caption_words), 'normMaps', images[j].split('.')[0])
            # Visualize heatmap tensor before and after downsampling
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(normalized_norm_map_tensor_image, cmap='gray')
            axs[0].set_title("norm_map tensor")

            axs[1].imshow(downsampled_norm_map_image, cmap='gray')
            axs[1].set_title("norm_map tensor downsampled")
            
            if save_normMap_path is not None:
                save_normMap_path_dir = os.path.dirname(save_normMap_path)
                if not os.path.exists(save_normMap_path_dir):
                    os.makedirs(save_normMap_path_dir)
                plt.savefig(save_normMap_path, bbox_inches='tight', pad_inches=0.1)
                plt.show()
            else:
                plt.show()
            
            plot_inference(im, caption, normalized_norm_map_tensor,save_fig_path)








# # MDETR_4obj_2mustards + Gaze info
# im = Image.open("/home/suka/code/mdetr/MDETR_test_data/MDETR_2mustards_biggerFront/rgb_img/00000012.ppm")
# im.show()

# # Raw heatmap from VTD
# raw_hm = torch.load("/home/suka/code/mdetr/raw_hm_dump(1)/raw_hm_dump/raw_hm_MDETR_4obj_2mustards/tensor1704893612.pt")
# normalized_raw_hm = raw_hm / torch.sum(raw_hm, dim=(2,3), keepdim=True)
# print("raw_hm shape: ", raw_hm.shape)
# print("normalized_raw_hm shape: ", normalized_raw_hm.shape)

# # Modulated heatmap from VTD
# norm_map = Image.open("/home/suka/code/mdetr/MDETR_test_data/MDETR_2mustards_biggerFront/normMap/00000009.ppm")
# #print("norm_map size: ",norm_map.size, "\n norm_map max", max(norm_map.getdata()))

# Modulated heatmap from VTD - Gray scale
#norm_map_gray = norm_map.convert('L')
#print("norm_map_gray size: ",norm_map_gray.size, "\n norm_map_gray max", max(norm_map_gray.getdata()))

# norm_map RGB and Grayscale visualization
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].imshow(norm_map)
# axs[0].set_title("Original Heatmap")

# axs[1].imshow(norm_map_gray, cmap='gray')
# axs[1].set_title("Gray scale heatmap")
# plt.show()


# # norm_map normalize and resize
# normalized_norm_map = norm_map_gray.point(lambda x: (x - 127.5) / 127.5)
# transform_normMap_noTensor = T.transforms.Compose([
#     T.Resize(size=(25,34))
# ])
# reszied_normmap_image = transform_normMap_noTensor(normalized_norm_map)
# print("reszied_normmap_image max: ", max(reszied_normmap_image.getdata()), "\n reszied_normmap_image shape: ", reszied_normmap_image.size)

# # resized image to tensor
# transform_normMap_image_Tensor = T.transforms.Compose([
#     T.transforms.ToTensor()
# ])
# reszied_normmap_image_tensor = transform_normMap_image_Tensor(reszied_normmap_image)
# reszied_normmap_image_tensor_array = np.squeeze((reszied_normmap_image_tensor.cpu().numpy())*255)
# print("normalized_norm_map_tensor_array shape", reszied_normmap_image_tensor.shape)
# reszied_normmap_image_tensor_image = Image.fromarray(reszied_normmap_image_tensor_array.astype(np.uint8))
# reszied_normmap_image_tensor_image.show()


#normalized_norm_map_tensor = transform_normMap(norm_map_gray)
#print("normalized_norm_map_tensor max: ", torch.max(normalized_norm_map_tensor), "\n normalized_norm_map_tensor min: ", torch.min(normalized_norm_map_tensor), "\n normalized_norm_map_tensor shape: ", normalized_norm_map_tensor.shape, "\n",normalized_norm_map_tensor)

# # Visualize norm_map tensor before downsampling
# normalized_norm_map_tensor_array = np.squeeze(normalized_norm_map_tensor.cpu().numpy())*255
# #print("normalized_norm_map_tensor_array shape", normalized_norm_map_tensor_array.shape)
# normalized_norm_map_tensor_image = Image.fromarray(normalized_norm_map_tensor_array.astype(np.uint8))

# # Visualize downsampled norm map
# downsampled_norm_map = torch.nn.functional.interpolate(normalized_norm_map_tensor.unsqueeze(0),size=(25,34), mode='bilinear', align_corners=False).squeeze(0)
# print("downsampled_norm_map max: ", torch.max(downsampled_norm_map), "\n downsampled_norm_map shape: ", downsampled_norm_map.shape)
# downsampled_norm_map_array = np.squeeze(downsampled_norm_map.cpu().numpy())*255
# downsampled_norm_map_image = Image.fromarray(downsampled_norm_map_array.astype(np.uint8))

# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].imshow(reszied_normmap_image, cmap='gray')
# axs[0].set_title("resized heatmap image")

# axs[1].imshow(downsampled_norm_map_image, cmap='gray')
# axs[1].set_title("downsampled heatmap tensor")
# plt.show()

# # Visualize heatmap tensor before and after downsampling
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].imshow(normalized_norm_map_tensor_image, cmap='gray')
# axs[0].set_title("norm_map tensor")

# axs[1].imshow(downsampled_norm_map_image, cmap='gray')
# axs[1].set_title("norm_map tensor downsampled")
# plt.show()


#plot_inference(im, "Pass the small yellow mustard bottle on the right.", normalized_norm_map_tensor)