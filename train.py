import argparse

import image_classifier as fn

parser = argparse.ArgumentParser(description='Train.py')

parser.add_argument('--data_dir', nargs='*', action="store", default="./flowers/",
                    metavar='', help="Define the directory for data ")

parser.add_argument('--gpu', dest="gpu", action="store", default="gpu",
                    metavar='', help="GPU training")

parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.005,
                    metavar='', help="Learning rate. Default = 0.001")

parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=5,
                    metavar='', help="Number of epochs. Default = 5")

parser.add_argument('--arch', dest="arch", action="store", default="vgg16_bn", type=str,
                    metavar='', help="CNN model architecture: vgg16_bn")

args = parser.parse_args()


trainloader, validloader, testloader, cat_to_name, train_data = fn.load_data(args.data_dir)
device, model, criterion, optimizer = fn.build_model(args.gpu, args.arch)
fn.train_model(model, criterion, optimizer, trainloader, validloader, args.epochs)
fn.test_model(model, testloader, device, criterion)
fn.save_checkpoint(args.epochs, 64, model, optimizer, criterion, train_data)

print("Done. Model was trained")