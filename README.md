# BreastAgeNet_MIL

This package aims to implement a multiple instance learning (MIL) framework to train and test on selected NBT patches from usual-risk women.

## Table of Contents
- Installation
- Usage
- Features
- Contact

## Installation
1. Clone the repository:

   git clone ....

2. Navigate to the project directory:

   cd BreastAgeNet_MIL
   
3. Create and activate a virtual environment:

   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

4. Install dependencies:

    pip install -r requirements.txt

## Usage

To classify NBT into epithelium, stroma, and adipocytes, run
    python main.py --mask_dir  --TC_dir  --patch_size

To localise lobules, create bounding box (bbx) and visualise bbx on the WSI, run:
    python main.py --TC_dir  --bbx_npy   --bbx_vis





    