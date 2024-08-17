import os
import cv2
import csv
import glob
import time
import shutil
import numpy as np
import pandas as pd

import openslide
import staintools
from skimage import morphology
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000


# Tensorflow Dependencies
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.layers import Resizing



##########################################################################################################
# for task run_TC
##########################################################################################################
def get_wsiname(file):
    if ".mrxs" in file:
        slidename = os.path.basename(file).split(".mrxs")[0]
    elif ".ndpi" in file:
        slidename = os.path.basename(file).split(".ndpi")[0]
    elif ".svs" in file:
        slidename = os.path.basename(file).split(".svs")[0]
    return slidename


# given a fixed patch size (um), calculate the pixel size on 40x WSI
def parse_patch_size(wsi, patch_size=128):
    mpp_x, mpp_y = float(wsi.properties['openslide.mpp-x']), float(wsi.properties['openslide.mpp-y'])
    patch_size_x, patch_size_y = int(patch_size // mpp_x), int(patch_size // mpp_y)
    return patch_size_x, patch_size_y


def Reinhard(img_arr):
    standard_img = "/scratch/prj/cb_histology_data/Siyuan/he_shg_synth_workflow/thumbnails/he.jpg"
    # the standard_img was downloaded from https://github.com/uw-loci/he_shg_synth_workflow/tree/v0.1.0
    target = staintools.read_image(standard_img)
    target = staintools.LuminosityStandardizer.standardize(target)
    normalizer = staintools.ReinhardColorNormalizer()
    normalizer.fit(target)
    #img = staintools.read_image(img_path)
    img_to_transform = staintools.LuminosityStandardizer.standardize(img_arr)
    img_transformed = normalizer.transform(img_to_transform)
    return img_transformed


# get TC model
def get_TC(weights):
    IMAGE_SIZE = (512,512)
    NUM_CLASSES = 3
    initializer = tf.keras.initializers.GlorotNormal()
    net = MobileNet(include_top=False, input_tensor=None,input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
    x = net.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', name = 'Dense_1', kernel_initializer=initializer)(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', name = 'Dense_2', kernel_initializer=initializer)(x)
    output_layer = Dense(NUM_CLASSES, activation='softmax', name='Predictions')(x)
    net_final = Model(inputs=net.input, outputs=output_layer,trainable=False)
    net_final.load_weights(weights)
    return net_final
    


# yield patches from WSI as inputs for TC
class SlideIterator(object):
    '''
    This code was modified based on https://github.com/davidtellez/neural-image-compression/tree/master
    '''
    def __init__(self, wsi, image_level, mask_path=None, threshold_mask=0.8):
        self.image = wsi
        self.image_level = image_level
        self.image_shape = self.image.level_dimensions[image_level]
        self.image_level_multiplier = self.image.level_dimensions[0][0] // self.image.level_dimensions[1][0]
        self.mask_path = mask_path
        self.mask = np.transpose(np.load(mask_path), axes=(1, 0))
        self.mask_shape = self.mask.shape
        if self.image_shape != self.mask_shape:
            self.image_mask_ratio = int(self.image_shape[0] / self.mask_shape[0])
        else:
            self.image_mask_ratio = 1
        self.threshold_mask = threshold_mask # use 0.3 for BCI lighter color WSIs

    def get_image_shape(self, stride):
        if self.image_shape is not None:
            return (self.image_shape[0] // stride, self.image_shape[1] // stride)
        else:
            return None

    def iterate_patches(self, patch_size, stride, downsample=1):
        self.feature_shape = self.get_image_shape(stride)
        for index_y in range(0, self.image_shape[1], stride):
            for index_x in range(0, self.image_shape[0], stride):
                # Avoid numerical issues by using the feature size
                if (index_x // stride >= self.feature_shape[0]) or (index_y // stride >= self.feature_shape[1]):
                    continue
                # use the foreground mask to filter patches
                if self.mask_path is not None:
                    mask_x, mask_y = int(index_x / self.image_mask_ratio), int(index_y / self.image_mask_ratio)
                    mask_tile_plt = self.mask[mask_x: (mask_x + patch_size // self.image_mask_ratio), mask_y:(mask_y + patch_size // self.image_mask_ratio)]
                    mask_tile = mask_tile_plt.flatten()
                else:
                    mask_tile = None
                # Continue only if it is full of tissue.
                if (mask_tile is None) or (mask_tile.mean() > self.threshold_mask):
                    image_tile = np.array(
                        self.image.read_region((int(index_x * (self.image_level_multiplier ** self.image_level)),
                                                int(index_y * (self.image_level_multiplier ** self.image_level))),
                                               self.image_level,
                                               (patch_size, patch_size)).convert("RGB")).astype('uint8')
                    # reinhard normalise tiles
                    image_tile = Reinhard(image_tile)
                    # Downsample
                    if downsample != 1:
                        image_tile = image_tile[::downsample, ::downsample, :]
                    if (mask_tile is None) and image_tile.mean() >= 240:
                        continue
                    # Yield mask_path, 
                    yield (image_tile, mask_tile_plt, index_x // stride, index_y // stride)

    def save_array(self, patch_size, stride, output_pattern, downsample=1):
        filename = os.path.splitext(os.path.basename(output_pattern))[0]

        image_tiles = []
        xs = []
        ys = []
        for image_tile, _, x, y in self.iterate_patches(patch_size, stride, downsample=downsample):
            image_tiles.append(image_tile)
            xs.append(x)
            ys.append(y)

        image_tiles = np.stack(image_tiles, axis=0).astype('uint8')
        xs = np.array(xs)
        ys = np.array(ys)
        image_shape = self.get_image_shape(stride)

        np.save(f"{output_pattern}_patches.npy", image_tiles)
        np.save(f"{output_pattern}_x_idx.npy", xs)
        np.save(f"{output_pattern}_y_idx.npy", ys)
        np.save(f"{output_pattern}_im_shape.npy", image_shape)

        print(f"{self.image.level_dimensions[0]}, {image_shape} compressed, in which {image_tiles.shape[0]} tissue tiles.\nTiles are extracted at level {self.image_level}, with a size of {patch_size} \nand a stride of {stride} and downsampled with a factor of {downsample}.")



# prepare dataset that yield batches for TC prediction
class WsiNpySequence(keras.utils.Sequence):
    '''
    This code was modified based on https://github.com/davidtellez/neural-image-compression/tree/master
    '''
    def __init__(self, wsi_pattern, batch_size):
        # Params
        self.batch_size = batch_size
        self.wsi_pattern = wsi_pattern
        # Read data
        self.image_tiles = np.load(wsi_pattern + '_patches.npy')
        self.xs = np.load(wsi_pattern + '_x_idx.npy')
        self.ys = np.load(wsi_pattern + '_y_idx.npy')
        self.image_shape = np.load(wsi_pattern + '_im_shape.npy')
        self.n_samples = self.image_tiles.shape[0]
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))

    def __len__(self):
        return self.n_batches

    def get_batch(self, idx):
        # Get samples
        idx_batch = idx * self.batch_size
        if idx_batch + self.batch_size >= self.n_samples:
            idxs = np.arange(idx_batch, self.n_samples)
        else:
            idxs = np.arange(idx_batch, idx_batch + self.batch_size)

        # Check patch size & Build batch
        if self.image_tiles[0].shape == (512, 512):
            image_tiles = self.image_tiles[idxs, ...]
        else:
            image_tiles = self.image_tiles[idxs, ...]
            image_tiles = Resizing(512,512)(image_tiles)

        # Format
        image_tiles = preprocess_input(image_tiles)
        return image_tiles

    def __getitem__(self, idx):
        batch = self.get_batch(idx)
        return batch



def run_TC_one_slide(wsi, mask_pt, TC, temp, patch_size=128, foreground_thes=0.7):
    mask_arr = np.array(Image.open(mask_pt))
    mask_path = mask_pt.replace(".png", ".npy")
    np.save(f"{temp}/{os.path.basename(mask_path)}", (mask_arr/255).astype("uint8"))
        
    time1 = time.time()
    si = SlideIterator(wsi=wsi, image_level=0, mask_path=mask_path, threshold_mask=foreground_thes)
    patch_size, _ = parse_patch_size(wsi, patch_size)
    si.save_array(patch_size=patch_size, stride=patch_size, output_pattern=temp, downsample=1)
    print(f"{time.time() - time1} seconds to vectorise the slide!")
    print(f"Tissue-Classifier is predicting ...")
    wsi_sequence = WsiNpySequence(wsi_pattern=temp, batch_size=8)
    tc_predictions = TC.predict_generator(generator=wsi_sequence, steps=len(wsi_sequence),verbose=1)
                
    xs = wsi_sequence.xs
    ys = wsi_sequence.ys
    image_shape = wsi_sequence.image_shape
    tissue_map = np.ones((image_shape[1], image_shape[0], tc_predictions.shape[1])) * np.nan
                
    for patch_feature, x, y in zip(tc_predictions, xs, ys):
        tissue_map[y, x, :] = patch_feature
    tissue_map[np.isnan(tissue_map)] = 0
    return tissue_map


def TC_pred(crop_norm, TC):
    input_img = np.expand_dims(np.array(crop_norm), axis=0)
    input_img = tf.keras.applications.mobilenet.preprocess_input(input_img)
    TC_epi, TC_str, TC_adi = TC.predict(input_img)[0]
    return TC_epi, TC_str, TC_adi


# plot the WSI TC map
def plot_TCmap(tissue_map, require_bounds=False):
    tissue_map = np.load(TC_maskpt)
    if require_bounds:
        bounds_h = int(wsi.properties['openslide.bounds-height'])//512
        bounds_w = int(wsi.properties['openslide.bounds-width'])//512
        bounds_x = int(wsi.properties['openslide.bounds-x'])//512
        bounds_y = int(wsi.properties['openslide.bounds-y'])//512
        tissue_map = tissue_map[bounds_y:(bounds_y+bounds_h), bounds_x:(bounds_x+bounds_w),:]
    im = Image.fromarray((tissue_map * 255).astype("uint8"))
    return im
     

##########################################################################################################
# for task run_bbx
##########################################################################################################

# modify TC mask to get connected epithelial objects
def process_TCmask(wsi_pt, TC_maskpt, upsample, small_objects, roi_width):
    wsi = openslide.OpenSlide(wsi_pt)
    mask_TC = np.load(TC_maskpt)
    
    # calculate the size of small_objects on the mask
    wsi_mask_ratio = wsi.level_dimensions[0][0] / mask_TC.shape[1] / upsample
    small_objects = small_objects / wsi_mask_ratio
    # calculate the size of width between inner and outer bbxes on the mask
    mpp = float(wsi.properties['openslide.mpp-x'])
    roi_width = int(roi_width / float(mpp) / wsi_mask_ratio) 
    
    # get a bigger epithelium mask
    epi_mask = mask_TC.copy()
    epi_mask = np.argmax(epi_mask, axis=-1)
    epi_mask[mask_TC[:,:,0]==0] = 99
    epi_mask[epi_mask != 0] = 99
    epi_mask[epi_mask == 0] = 1
    epi_mask[epi_mask == 99] = 0
    epi_mask = cv2.resize(np.uint8(epi_mask * 255), (epi_mask.shape[1]*upsample, epi_mask.shape[0]*upsample), interpolation=cv2.INTER_CUBIC) 
    epi_mask = np.array(epi_mask) > 0 
    
    # remove object area less than small_objects
    epi_mask = morphology.remove_small_objects(epi_mask.astype(bool), small_objects) 
    # fill small holes
    epi_mask = morphology.remove_small_holes(epi_mask.astype(bool), small_objects)
    epi_mask = epi_mask.astype("uint8")

    return epi_mask, roi_width, wsi_mask_ratio



# detect a single ROI
# defined as the bounding box of a connected epithelium object which is further expanded with a certain distance
def get_roi_locs(mask, roi_width):
    roi = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    roi[:,:,0][mask] = np.random.randint(0,255)
    roi[:,:,1][mask] = np.random.randint(0,255)
    roi[:,:,2][mask] = np.random.randint(0,255)

    y_idx, x_idx, _ = np.nonzero(roi) 
    x1_inner, x2_inner = np.min(x_idx), np.max(x_idx)
    y1_inner, y2_inner = np.min(y_idx), np.max(y_idx)

    x1_outer = np.max((0, x1_inner-roi_width))
    y1_outer = np.max((0, y1_inner-roi_width))
    x2_outer = np.min((x2_inner+roi_width, roi.shape[1]))
    y2_outer = np.min((y2_inner+roi_width, roi.shape[0]))
    
    return x1_inner, y1_inner, x2_inner, y2_inner, x1_outer, y1_outer, x2_outer, y2_outer



# save a png file showing ROI overlay on the WSI
def bbx_overlay(epi_mask, overlay_pt, roi_width):    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(epi_mask, connectivity=8)
    
    bbx_map = np.zeros((epi_mask.shape[0], epi_mask.shape[1], 3), np.uint8)
    for i in range(1, num_labels):
        mask_i = labels == i
        bbx_map[:,:,0][mask_i] = np.random.randint(0,255)
        bbx_map[:,:,1][mask_i] = np.random.randint(0,255)
        bbx_map[:,:,2][mask_i] = np.random.randint(0,255)
        
        x1_inner, y1_inner, x2_inner, y2_inner, x1_outer, y1_outer, x2_outer, y2_outer = get_roi_locs(mask_i, roi_width)
        # avoid getting too large ROIs
        if (x2_outer-x1_outer < mask_i.shape[1]//5) and (y2_outer-y1_outer < mask_i.shape[0]//5):
            bbx_map = cv2.rectangle(bbx_map, (x1_inner, y1_inner), (x2_inner,y2_inner), (255, 0, 0), 15)
            bbx_map = cv2.rectangle(bbx_map, (x1_outer, y1_outer), (x2_outer, y2_outer), (255, 255, 0), 15)

    plt.imshow(bbx_map)
    plt.axis("off")
    plt.savefig(overlay_pt, bbox_inches='tight', pad_inches=0)
    return bbx_map


    
# get roi ids for all ROIs detected on the WSI
def get_roi_ids(epi_mask, wsi_id, roi_width, upsample, wsi_mask_ratio):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(epi_mask, connectivity=8)
    roi_ids = []
    for i in range(1, num_labels):
        mask_i = labels == i
        _,_,_,_, x1_outer, y1_outer, x2_outer, y2_outer = get_roi_locs(mask_i, roi_width)

        # avoid getting too large ROIs for NKI cohort
        threshold = 3
        if (x2_outer-x1_outer < mask_i.shape[1]//threshold) and (y2_outer-y1_outer < mask_i.shape[0]//threshold):
            x1_outer, y1_outer, x2_outer, y2_outer =x1_outer*wsi_mask_ratio, y1_outer*wsi_mask_ratio, x2_outer*wsi_mask_ratio, y2_outer*wsi_mask_ratio
            roi_id = f"{wsi_id}_{int(x1_outer)}_{int(y1_outer)}_{int(x2_outer-x1_outer)}_{int(y2_outer-y1_outer)}"
            roi_ids.append(roi_id)
            
    print(f"There're {len(np.unique(roi_ids))} ROIs detected")
    return roi_ids



def roi_id2patch_id(roi_id, patch_size, TC_maskpt, patch_dict):
    # parse roi id
    roi_x, roi_y, roi_w, roi_h = roi_id.split("_")[-4:]
    roi_x, roi_y, roi_w, roi_h = int(roi_x), int(roi_y), int(roi_w), int(roi_h)

    # check roi TC grids
    grid_x1 = int(np.ceil(roi_x/patch_size))
    grid_y1 = int(np.ceil(roi_y/patch_size))
    grid_x2 = int(np.floor(roi_w/patch_size)) + grid_x1
    grid_y2 = int(np.floor(roi_h/patch_size)) + grid_y1


    TC_mask = np.load(TC_maskpt)
    roi_TC = TC_mask[grid_y1:grid_y2, grid_x1:grid_x2]
    # plt.imshow(roi_TC)
    
    roi_TC_cls = np.zeros((roi_TC.shape[0], roi_TC.shape[1])) 
    roi_TC_cls[roi_TC[:,:,0] > 0.5] = 1 # epi
    roi_TC_cls[roi_TC[:,:,1] > 0.5] = 2 # str

    # plt.imshow(roi_TC_cls)
    cls_dict = {"0": "adipocytes", "1": "epithelium", "2": "stroma"}
    for cls in [0,1,2]:   
        ys, xs = np.nonzero(roi_TC_cls == cls) 
        for y, x in zip(ys, xs):
            # wsi grids
            grid_y, grid_x = y+grid_y1, x+grid_x1
            patch_id = f"{roi_id}_{grid_x}_{grid_y}_{patch_size}"

            patch_dict["roi_id"].append(roi_id)
            patch_dict["patch_id"].append(patch_id)
            patch_dict["cls"].append(cls_dict[str(cls)])
            
            patch_dict["TC_epi"].append(TC_mask[grid_y, grid_x, 0])
            patch_dict["TC_str"].append(TC_mask[grid_y, grid_x, 1])
            patch_dict["TC_adi"].append(TC_mask[grid_y, grid_x, 2])




def save_patchcsv(roi_ids, patch_size, TC_maskpt, output_dir):
    patch_dict = {"roi_id": [], "patch_id": [], "cls": [], "TC_epi": [], "TC_str": [], "TC_adi": []}
    for roi_id in roi_ids:
        roi_id2patch_id(roi_id, patch_size, TC_maskpt, patch_dict)

    patch_df = pd.DataFrame.from_dict(patch_dict)
    patch_df = patch_df.loc[patch_df['cls'] != "adipocytes", :]
    patch_df["cohort"] = os.path.basename(output_dir)
    patch_df["wsi_id"] = os.path.basename(output_dir)

    patch_csv = f"{output_dir}/{wsi_id}_patch.csv"
    patch_df.to_csv(patch_csv, index=False)
    print(f"{patch_csv} saved!")

    return patch_df


def parse_roi_id(roi_id):
    wsi_id = "_".join(roi_id.split("_")[:-4])
    x,y,w,h = int(roi_id.split("_")[-4]), int(roi_id.split("_")[-3]), int(roi_id.split("_")[-2]), int(roi_id.split("_")[-1])
    return wsi_id, x, y, w, h
    

# save roi images
def show_roi(wsi, roi_id):
    wsi_id, x_orig, y_orig, wid_orig, heigh_orig = parse_roi_id(roi_id)
    roi_img = wsi.read_region((int(x_orig), int(y_orig)), 0, (int(wid_orig), int(heigh_orig))).convert("RGB")
    im = Image.fromarray(np.array(roi_img))
    return im


def parse_patch_id(patch_id):
    _,_,_,_,grid_x,grid_y,patch_size = patch_id.split('_')[-7:]
    orig_x = int(int(grid_x) * int(patch_size))
    orig_y = int(int(grid_y) * int(patch_size))
    return orig_x, orig_y, patch_size

    
def show_patch(wsi, patch_id):
    orig_x, orig_y, patch_size = parse_patch_id(patch_id)
    img = wsi.read_region((int(orig_x), int(orig_y)), 0, (int(patch_size), int(patch_size))).convert("RGB")
    plt.imshow(img)





