import os
os.chdir("/scratch/users/k21066795/prj_normal/awesome_normal_breast/scripts")
import glob
import numpy as np
import pandas as pd
import cv2
import openslide
from PIL import Image
import matplotlib.pyplot as plt
from utils.preprocessing import parse_patch_size,process_TCmask,bbx_overlay,get_roi_ids,get_patch_dict,roi_id2patch_id



def run_bbx(args):
    for wsi_id in os.listdir(args.TC_output):
        print(wsi_id)
        output_dir=f"{args.TC_output}/{wsi_id}"
        try:
            TC_maskpt = glob.glob(f"{output_dir}/{wsi_id}_TC*mask.npy")[0] 
        except:
            f"TC mask for {wsi_id} does not exists!"
            continue
            
        wsi_pt = glob.glob(f"{args.WSI}/{wsi_id}*.*")[0] 
        wsi = openslide.OpenSlide(wsi_pt)
        patch_size, _ = parse_patch_size(wsi, patch_size=args.patch_size)
            
        # get epithelium mask
        epi_mask, roi_width, wsi_mask_ratio = process_TCmask(wsi_pt, TC_maskpt, args.upsample, args.small_objects, args.roi_width)
            
        # ROI visualisation
        if args.save_bbxpng:
            overlay_pt = f"{output_dir}/{wsi_id}_bbx.png"
            if not os.path.exists(overlay_pt): 
                bbx_map = bbx_overlay(epi_mask, overlay_pt, roi_width)
                print(f"{overlay_pt} saved!")

        if args.save_patchcsv:
            roi_ids = get_roi_ids(epi_mask, wsi_id, roi_width, upsample, wsi_mask_ratio)
            save_patchcsv(roi_ids, patch_size, TC_maskpt, output_dir)
        


