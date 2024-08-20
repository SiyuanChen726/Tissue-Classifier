import os
import glob
import numpy as np
import time
import random
import openslide
from utils import get_wsiname, Reinhard, SlideIterator, get_TC, WsiNpySequence, parse_patch_size


def run_TC(args):
    foreground_masks = glob.glob(f"{args.MASK}/*/*_mask_use.png")
    # foreground_masks = [i for i in foreground_masks if " HE" not in i]
    # foreground_masks = [i for i in foreground_masks if "_FPE_" not in i]
    random.shuffle(foreground_masks)
    for mask_pt in foreground_masks:
        mask_arr = np.array(Image.open(mask_pt))
        mask_path = mask_pt.replace(".png", ".npy")
        np.save(mask_path, (mask_arr/255).astype("uint8"))
    
        wsiname = get_wsiname(mask_pt)
        # wsiname = os.path.basename(mask_pt).split(".svs")[0]
        print(f"preprocessing {wsiname}")
        
        wsi_pt = glob.glob(f"{args.WSI}/{wsiname}*.*")[0]
        print(wsi_pt)
        
        output_dir=f"{args.TC_output}/{wsiname}"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            
        if not os.path.exists(f"{output_dir}/{wsiname}_TCmask.npy"):
            try:
                wsi=openslide.OpenSlide(wsi_pt)
                print(f"vectorise {wsiname}")
            except:
                print(f"Openslide can not open {wsiname}")
                break
                
            output_pattern = os.path.join(output_dir, f"{wsiname}_pattern")
            time1 = time.time()
            si = SlideIterator(wsi=wsi, image_level=0, mask_path=mask_path, threshold_mask=args.foreground_thes)
            
            patch_size, _ = parse_patch_size(wsi, args.patch_size)
            
            si.save_array(patch_size=patch_size, stride=patch_size, output_pattern=output_pattern, downsample=1)
            print(f"{time.time() - time1} seconds to vectorise {wsiname}!")
            
            print(f"Tissue-Classifier is predicting {wsiname}")
            tissue_classifier=get_TC(args.WEIGHT)
            wsi_sequence = WsiNpySequence(wsi_pattern=output_pattern, batch_size=8)
            tc_predictions = tissue_classifier.predict_generator(generator=wsi_sequence, steps=len(wsi_sequence),verbose=1)
            
            xs = wsi_sequence.xs
            ys = wsi_sequence.ys
            image_shape = wsi_sequence.image_shape
            tissue_map = np.ones((image_shape[1], image_shape[0], tc_predictions.shape[1])) * np.nan
            
            for patch_feature, x, y in zip(tc_predictions, xs, ys):
                tissue_map[y, x, :] = patch_feature
            tissue_map[np.isnan(tissue_map)] = 0
            npy_pt = f"{output_dir}/{wsiname}_TCprobmask.npy"
            np.save(npy_pt, tissue_map)

            print(f"saved!")

            # save tissue classification heatmap
            if args.save_TCmap:
                if ".mrxs" in os.path.basename(wsi_pt):
                    bounds_h = int(wsi.properties['openslide.bounds-height'])//patch_size
                    bounds_w = int(wsi.properties['openslide.bounds-width'])//patch_size
                    bounds_x = int(wsi.properties['openslide.bounds-x'])//patch_size
                    bounds_y = int(wsi.properties['openslide.bounds-y'])//patch_size
                    tissue_map = tissue_map[bounds_y:(bounds_y+bounds_h), bounds_x:(bounds_x+bounds_w),:]
            
                im = Image.fromarray((tissue_map * 255).astype("uint8"))
                im.save(f"{output_dir}/{wsiname}_TC_({patch_size},0,0,{image_shape[0]},{image_shape[1]}).png")
            
            if args.free_space:
                for i in glob.glob(f"{output_dir}/*pattern*"):
                    os.remove(i)
                    print(f"{i} removed!")





