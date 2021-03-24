import argparse

import image_classifier as fn

parser = argparse.ArgumentParser(description='Train.py')

parser.add_argument('-dir','--data_dir', type=str, action="store", default="./flowers",
                    metavar='', help="Define the directory of the data ")
parser.add_argument('-gpu','--processor', dest="processor", action="store", default="gpu", type=str, 
                    metavar='', help="GPU training")
parser.add_argument('-lr', '--learning_rate', dest="learning_rate", action="store", default=0.005, type=float,
                    metavar='', help="Learning rate. Default = 0.001")
parser.add_argument('-ep', '--epochs', dest="epochs", action="store", type=int, default=8,
                    metavar='', help="Number of epochs. Default = 5")
parser.add_argument('--arch', dest="arch", action="store", choices=["vgg19_bn", "densenet121"], default="vgg19_bn", type=str,
                    metavar='', help="CNN model architecture: vgg19_bn or densenet121")
parser.add_argument('-hs','--hidden_sizes', dest="hidden_sizes", action="store", type=int,
                    metavar='', help="Size of hidden layer of model classifier")
parser.add_argument('-bs', '--batch_size', dest="batch_size", action="store", type=int, default=64, 
                    metavar='', help="Size of batches")
parser.add_argument('-cat', '--category_names', type= str, dest="category_names", action="store", default='cat_to_name.json',
                    metavar='', help="Define The category name")

args = parser.parse_args()


image_datasets, dataloaders = fn.load_data(args.data_dir, args.batch_size)
model, criterion, optimizer = fn.build_model(args.arch, args.hidden_sizes, args.processor, args.learning_rate, args.category_names)
fn.train_model(model, criterion, optimizer, dataloaders['trainloader'], dataloaders['validloader'], args.epochs, args.processor)
fn.test_model(model, dataloaders['testloader'], criterion, args.processor)
fn.save_checkpoint(args.arch, args.learning_rate, model, optimizer, criterion, image_datasets, args.hidden_sizes)