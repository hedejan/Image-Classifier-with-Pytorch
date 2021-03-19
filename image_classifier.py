
import torch
from torch import nn
from torch import optim
import numpy as np
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
from PIL import Image
import torch.utils.data


batch_size =64
# Train
def load_data(data_dir="./flowers"):
    """
    Receives the location of the image files,
    applies the necessary transformations,
    converts the images to tensor in order to be able to be fed into the neural network
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    # label mapping
    with open("cat_to_name.json", "r") as f:
        cat_to_name = json.load(f)

    print("================- Done || Loading || Data -================")

    return trainloader, validloader, testloader, cat_to_name, train_data


def load_classifier():
    """
    :return: vgg16_bn classifier parameters
    """
    input_size = 25088
    hidden_sizes = 1024
    output_size = 102

    classifier = nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(input_size, hidden_sizes)),
        ("relu1", nn.ReLU()),
        ("dropout1", nn.Dropout(0.2)),
        ("fc5", nn.Linear(hidden_sizes, output_size)),
        ("output", nn.LogSoftmax(dim=1))
    ]))
    return classifier


def build_model(power, arch):
    # use gpu model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define model
    if arch == "vgg16_bn":
        model = models.vgg16_bn(pretrained=True)
    else:
        print("Error: No model architecture defined!")

    # freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Specify classifier
    model.classifier = load_classifier()

    # specify loss function
    criterion = nn.NLLLoss()

    # specify optimizer
    opt = optim.Adam(model.classifier.parameters(), lr=0.005)

    # move the model to device
    model.to(device)

    print("================- Done || Model || Building -================")
    return device, model, criterion, opt


def train_model(model, criterion, optimizer, trainloader, validloader, epochs=5):
    """
    trains the model over a certain number of epochs,
    display the training, validation and accuracy
    """
    train_losses = []
    valid_losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(1, epochs + 1):
        # track training and validation-loss
        train_loss = 0
        valid_loss = 0
        accuracy = 0
        # model training
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        # model validation
        model.eval()
        with torch.no_grad():
            for images, labels in validloader:

                images, labels = images.to(device), labels.to(device)

                logps = model(images)

                loss = criterion(logps, labels)

                valid_loss += loss.item() * images.size(0)
                # Calculate accuracy
                ps = torch.exp(logps)
                top_ps, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        # calculate average losses and validation accuracy
        train_loss = train_loss/len(trainloader.sampler)
        valid_loss = valid_loss/len(validloader.sampler)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        validation_accuracy = accuracy/len(validloader)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.6f}'.format(
            epoch, train_loss, valid_loss, validation_accuracy))
    
    print("================- Training || Done -================")


def test_model(model, testloader, device, criterion):
    print("================- Training || Start -================")

    model.eval()  # disables dropout
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            logps = model(images)
            loss = criterion(logps, labels)
            test_loss += loss.item() * images.size(0)

            # Calculate accuracy
            ps = torch.exp(logps)
            top_ps, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            test_accuracy = accuracy/len(testloader)

        print('Test Accuracy of the model: {} %'.format(test_accuracy))

def save_checkpoint(epochs, batch_size, model, optimizer, criterion, train_dataset):
    """
    save trained model
    """
    checkpoint = {"model": models.vgg16_bn(pretrained=True),
                  "input_size": 25088,
                  "output_size": 102,
                  "epochs": 5,
                  "batch_size": 64,
                  "state_dict": model.state_dict(),
                  "state_features_dict": model.features.state_dict(),
                  "state_classifier_dict": model.classifier.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "criterion_state_dict": criterion.state_dict(),
                  "class_to_idx": train_dataset.class_to_idx,
                  }
    torch.save(checkpoint, 'checkpoint.pth')

    model.cpu()

    load_model = torch.load('checkpoint.pth')
    load_model.keys()

    print("================- Done || Saving || Model -================")


# Predict
def load_checkpoint(path_dir='checkpoint.pth'):
    """
    :param pth_dir: checkpoint directory
    :return: nn model prediction
    """
    device, model, criterion, optimizer = build_model('cpu', 'vgg16_bn')
    check = torch.load(path_dir, map_location='cpu')
    model.load_state_dict(check["state_dict"], strict=False)
    print("================- Done || Rebuild || Model -================")
    return model


def process_image(img_path):
    """
    :param img_path: The directory for image
    :return: Tensor image
    """
    img_path = str(img_path)
    img = Image.open(img_path)
    transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])
    tensor_img = transform(img)
    return tensor_img


def predict_class(cat_to_name, img_path, top_k=5, device="gpu"):
    """
     Predict the class of the image
    :param cat_to_name: flowers name
    :param img_path: The path to the image
    :param top_k: The numbers of predictions
    :param device: Use cpu vs gpu if available
    :return: top_k probability
    """

    model = load_checkpoint()
    model.eval()

    img = process_image(img_path)
    img = img.unsqueeze(0)

    with torch.no_grad():
        if model == 0:
            print("LoadCheckpoint: ERROR - Checkpoint load failed")
        else:
            print("LoadCheckpoint: Checkpoint loaded")

        if torch.cuda.is_available() and device == "gpu":
            device = 'cuda'
            model.to('cuda')
        else:
            device = 'cpu'
            model.to('cpu')

        inputs = img.to(device, dtype=torch.float)
        log_ps = model.forward(inputs)
        ps = torch.exp(log_ps)
        classes = ps.topk(top_k, dim=1)
    model.train()

    classes_ps = classes[0]
    classes_ps = classes_ps.cpu().tolist()
    classes_ps = [item for sublist in classes_ps for item in sublist]

    # Extract predicted class index, copy tensor to CPU, convert to list.
    classes_idx = classes[1]
    classes_idx = classes_idx.cpu().tolist()

    # Get predicted flower names from cat_to_name
    class_names = [cat_to_name.get(str(idx)) for idx in np.nditer(classes_idx)]

    print("Class Index: ", classes_idx)
    print("Class Names: ", class_names)
    print("Class Probabilities: ", classes_ps)

    return classes_ps, class_names, ps, classes

    print("================- -:) Done (:- -================")