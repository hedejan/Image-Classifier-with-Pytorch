
"""
Predict flower classes of an image using predict.py with probability distribution.
You select a single image /path/to/image and return the flower name and class probability.
Basic usage: python predict.py /path/to/image checkpoint
Options:
Return top_K most K likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu
"""

import argparse
import image_classifier as fn

parser = argparse.ArgumentParser(description="Predict Image Class")

parser.add_argument('-dir', '--data_dir', type= str, action="store", default="./flowers",
                    metavar='', help="Define the directory for data ")
parser.add_argument('-ch', '--checkpoint', type= str, default= 'checkpoint.pth',
                    action="store", metavar='', help="Define the directory to the checkpoint.pth file")
parser.add_argument('-tk', '--top_k', default=5, dest="top_k", action="store", type=int,
                    metavar='', help="To show the top k predictions")
parser.add_argument('-cat', '--category_names', type= str, dest="category_names", action="store", default='cat_to_name.json',
                    metavar='', help="Define The category name")
parser.add_argument('-gpu','--processor', dest="processor", action="store", default="cpu", type=str, 
                    metavar='', help="GPU training")

args = parser.parse_args()

fn.predict_class(args.checkpoint, args.processor, args.category_names, args.data_dir, args.top_k)