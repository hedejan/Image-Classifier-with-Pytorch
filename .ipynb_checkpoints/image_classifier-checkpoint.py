import torch, random, json
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
from os import listdir
from os.path import isfile, join
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
from torch.utils.data import DataLoader
from pprint import pprint

# Train
def load_data(data_dir, batch_size):
    """
    Receives the location of the image files,
    applies the necessary transformations,
    converts the images to tensor in order to be able to be fed into the neural network
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    data_transforms    = {
                     'train': transforms.Compose([
                            transforms.RandomRotation(30),
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ]),
                        'valid': transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ]),
                        'test': transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ]),
                    }

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    
    image_datasets = {'train': train_data,
                 'valid': valid_data,
                 'test': test_data,
                 }

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True)
    
    dataloaders = {'trainloader':trainloader, 
               'validloader': validloader, 
               'testloader': testloader,
              }
    
    print("================- Done || Loading || Data -================")

    return image_datasets, dataloaders


def load_classifier(arch, hidden_sizes, output_size):
    """
    :return: vgg19_bn or densenet121 classifier parameters
    """
    
    if arch == 'vgg19_bn':
        input_size = 25088
    else:
        input_size = 1024

    classifier = nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(input_size, hidden_sizes)),
        ("relu1", nn.ReLU()),
        ("dropout1", nn.Dropout(0.2)),
        ("fc5", nn.Linear(hidden_sizes, output_size)),
        ("output", nn.LogSoftmax(dim=1))
    ]))
    return classifier


def build_model(arch, hidden_sizes, processor, lr, category_names):

    # define model
    if arch == "vgg19_bn":
        model = models.vgg19_bn(pretrained=True)
    else:
        model = models.densenet121(pretrained=True)

    # freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Specify classifier
    with open(category_names, "r") as f:
        cat_to_name = json.load(f)
    output_size = len(cat_to_name)
    
    model.classifier = load_classifier(arch, hidden_sizes, output_size)

    # specify loss function
    criterion = nn.NLLLoss()

    # specify optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    # move the model to device
    if processor == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')
    model.to(device)

    print("================- Done || Building || Model -================")
    return model, criterion, optimizer


def train_model(model, criterion, optimizer, trainloader, validloader, epochs=8, processor='gpu'):
    """
    trains the model over a certain number of epochs,
    display the training, validation and accuracy
    """
    train_losses = []
    valid_losses = []
    if processor == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')
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
    
    print("================- Done || Training || Model -================")


def test_model(model, testloader, criterion, processor):
    
    if processor == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')
    model.to(device)
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
        
    print("================- Done || Testing || Model -================")

def save_checkpoint(arch, lr, model, optimizer, criterion, image_datasets, hidden_sizes):
    """
    save trained model
    """
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        "arch": arch,
        "hidden_sizes": hidden_sizes,
        "lr": lr,
        "model_state_dict": model.state_dict(),
        'model_classifier': model.classifier,
        "state_features_dict": model.features.state_dict(),
        "state_classifier_dict": model.classifier.state_dict(),
        "class_to_idx": model.class_to_idx,
        "optimizer_state_dict": optimizer.state_dict(),
        "criterion_state_dict": criterion.state_dict(),
    }
    torch.save(checkpoint, 'checkpoint.pth')

    print("================- Done || Saving || Model -================")


# Predict
def load_checkpoint(path_dir, processor, category_names):
    """
    :param pth_dir: checkpoint directory
    :return: nn model prediction
    """
    check = torch.load(path_dir, map_location='cpu')
    arch, hidden_sizes, lr = check['arch'], check['hidden_sizes'], check['lr']
    with open(category_names, "r") as f:
        cat_to_name = json.load(f)
    hidden_sizes = len(cat_to_name)
    model, criterion, optimizer = build_model(arch, hidden_sizes, processor, lr, category_names)
    if processor == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')
    model.to(device)
    model.classifier = check["model_classifier"]
    model.load_state_dict(check["model_state_dict"])
    optimizer = optim.Adam(model.classifier.parameters(), lr=check["lr"])
    optimizer.load_state_dict(check["optimizer_state_dict"])
    class_to_idx = check["class_to_idx"]
    print("================- Done || Rebuild || Model -================")
    return model, class_to_idx, device


def process_image(PATH):
    """
    :param img_path: The directory for image
    :return: Tensor image
    """
    img = Image.open(PATH)
    transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])
    tensor_img = transform(img)
    return tensor_img                             

def predict_class(path_dir, processor, category_names, data_dir, top_k=5):
    """
     Predict the class of the image
    :param cat_to_name: flowers name
    :param img_path: The path to the image
    :param top_k: The numbers of predictions
    :param device: Use cpu vs gpu if available
    :return: top_k probability
    """
                              
    model, class_to_idx, device = load_checkpoint(path_dir, processor, category_names)
    model.eval()
    
    
    with open(category_names, "r") as f:
        cat_to_name = json.load(f)
    # Get predicted flower names from cat_to_name
    idx_to_labels = {idx: cat_to_name[cat] for cat, idx in class_to_idx.items()}
    name_to_cat = {i:j for j, i in cat_to_name.items()}
                              
    label_idx = random.randint(0,101)
    label = idx_to_labels[label_idx]
    folder_num = name_to_cat[label]
    class_path = data_dir + "/test"+ f"/{folder_num}/"
    file_names = [f for f in listdir(class_path) if isfile(join(class_path, f))]
    PATH = class_path+file_names[random.randint(0,len(file_names)-1)]
    
    print(f"Class index: {label_idx}", 
           "==============",
          f"Class name: {label}",
           "==============",
          f"PATH: {PATH}")
    
    img = process_image(PATH)
    img = img.unsqueeze(0)

    with torch.no_grad():
        inputs = img.to(device, dtype=torch.float)
        output = model.forward(inputs)
    ps = F.softmax(output.data, dim=1)
    
    ps, classes = ps.topk(top_k, dim=1)
    np_ps = ps[0].cpu().numpy()
    np_classes = classes[0].cpu().numpy()

    class_names = {}
    for idx, class_number in enumerate(np_classes):
        class_names[class_number] = {idx_to_labels[class_number]: np_ps[idx]}

    pprint(class_names)

    print("================- -:) Done (:- -================")