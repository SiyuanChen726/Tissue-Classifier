from run_TC import run_TC
from run_bbx import run_bbx
import argparse


parser = argparse.ArgumentParser(description="Run Tissue-Classifier to get a 3-class mask")
parser.add_argument("--WSI", type=str, help="the folder storing foreground masks")
parser.add_argument("--MASK", type=str, help="the folder storing foreground masks")
parser.add_argument("--WEIGHT", type=str, help="the path of the weights for Tissue-Classifier")
parser.add_argument("--TC", type=str, help="the folder containing output files from running Tissue-Classifier")

# default
parser.add_argument("--foreground_thes", type=float, default=0.7, help="Only process patches with tissue area over this threshold")
parser.add_argument("--patch_size", type=int, default=128, help="the fix patch size (um)")
parser.add_argument("--save_TCmap", type=bool, default=True, help="Save .png file of the predicted Tissue type mask")
parser.add_argument("--free_space", type=bool, default=True, help="Remove the intermediate large files")

parser.add_argument("--save_bbxpng", type=bool, default=True, help="whether localise perilobular regions and save a .png file")
parser.add_argument("--save_patchcsv", type=bool, default=True, help="whether localise perilobular regions and save a .csv file storing patches classes and coordinates")
parser.add_argument("--upsample", action="int", default=32, help="upsample the tissue type mask {upsample} times")
parser.add_argument("--small_objects", action="int", default=400000, help="small objects/holes under this pixel size (at 40x magnification) will be removed or filled, for example, 400000 pixels at 40x are approximately 1.5 patches of 512x512 pixels")
parser.add_argument("--roi_width", action="int", default=250, help="the width (um) of the peri-lobular regions to include")


args = parser.parse_args()


if __name__ == "__main__":
    run_TC(args) 
    if any([args.save_bbxpng, args.save_bbxcsv, args.save_patchcsv])
        run_bbc(args)








        
