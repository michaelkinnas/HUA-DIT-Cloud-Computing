from torch.utils.data import DataLoader, Subset
from torch import cuda, inference_mode, argmax, save, manual_seed, backends, hub
from torch.nn import CrossEntropyLoss
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.datasets import CIFAR10
from torch.optim import Adam
from torch.hub import load
from torchvision.models import get_model
from utils.models_lists import imagenet_models
from utils.trainer import Trainer
from utils.datasets import ImageNetVal
from tqdm.auto import tqdm
from argparse import ArgumentParser


def measure_accuracy(model, dataloader, device, model_name):
    model.eval()
    acc = 0
    with inference_mode():
        for i, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Measuring accuracy of {model_name} on subset"):
            X, y = X.to(device), y.to(device)
            preds = model(X)
            acc += (argmax(preds, dim=1) == y).sum().item()
    return acc

def gather_wrong_preds(model, dataloader, device, batch_size, model_name):
    model.eval()
    indices = []
    with inference_mode():
        for i, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Getting incorrect predictions from {model_name}"):
            X, y = X.to(device), y.to(device)
            preds = model(X)
            for j, (pred, true) in enumerate(zip(preds, y)):
                if argmax(pred).item() != true.item():
                    indices.append(i * batch_size + j)
    return indices


def gather_correct_preds(model, dataloader, device, batch_size, model_name):
    model.eval()
    indices = []
    with inference_mode():
        for i, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Getting correct predictions from {model_name}"):
            X, y = X.to(device), y.to(device)
            preds = model(X)
            for j, (pred, true) in enumerate(zip(preds, y)):
                if argmax(pred).item() == true.item():
                    indices.append(i * batch_size + j)
    return indices


def load_model(model_name):
    if model_name in imagenet_models:
        return get_model(model_name, weights=imagenet_models[model_name])
    else:
        return load("chenyaofo/pytorch-cifar-models", model=model_name, pretrained=True)


def main():
    parser = ArgumentParser()
    parser.add_argument("-m1", "--model1", help="The first model, required. This parameter will set which dataset to use (CIFAR10 or ImageNet)", required=True)   
    parser.add_argument("-m2", "--model2", help="The second model.", default=None, required=False)
    parser.add_argument("-d", "--data-path", help="The path where the datasets (CIFAR-10 or custom ImageNet) are.")
    parser.add_argument("-e1", "--epochs-model1", help="The number of epochs that each fine-tuning step will take for the first model.", default=1, type=int)
    parser.add_argument("-e2", "--epochs-model2", help="The number of epochs that each fine-tuning step will take for the second model.", default=1, type=int)
    parser.add_argument("-b", "--batch-size", help="The size of each batch", default=32, type=int)
    parser.add_argument("-m", "--method", help="The fine-tune method to be used: Options: [1: Incorrect predictions from the other model. 2: Incorrect predictions from the other model plus correct predictions from the same model]", choices=['1', '2'])
    parser.add_argument("-ft1", "--fine_tune_first", help="If set the first model will be fine tuned from predictions of the second model.", default=False, action="store_true")
    parser.add_argument("-s", "--seed", help="The seed to be used in all pytorch classes.", default=None)    
    args = parser.parse_args()

    # Constants
    DATA_FILEPATH = args.data_path
    DATASET = "IMAGENET" if args.model1 in imagenet_models else "CIFAR10"

    device = 'cuda' if cuda.is_available() else 'cpu'
    if device == 'cuda':
        cuda.empty_cache()

    # Set manual seed for reproducability
    if args.seed is not None:
        manual_seed(args.seed)
        if device == 'cuda':
            cuda.manual_seed(args.seed)
            cuda.manual_seed_all(args.seed)
            backends.cudnn.benchmark = False
            backends.cudnn.deterministic = True

    if args.model1 in imagenet_models:
        print("Using ImageNet dataset")
        if args.model1 == 'inception_v3':
            transform = Compose([
                Resize(299),
                ToTensor(),
                Normalize(mean =(0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) # imagenet
            ])
        else:
            transform = Compose([
                # Resize(224),
                ToTensor(),
                Normalize(mean =(0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) # imagenet
            ])
        train_dataset = ImageNetVal(image_path=f'{DATA_FILEPATH}/ImageNet/val_resized_40K', transform=transform, random_seed=42)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        print("Using CIFAR10 dataset")
        transform = Compose([
                ToTensor(),
                Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201)) # CIFAR10
        ])
        train_dataset = CIFAR10(root=DATA_FILEPATH, download=True, train=True, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Load first model
    if DATASET == 'IMAGENET':
        model = get_model(args.model1, weights=imagenet_models[args.model1]).to(device)
    else:
        model = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{args.model1}', pretrained=True).to(device)

    # Get incorrect predictions from first model from proper dataset
    wrong_indices = gather_wrong_preds(model, train_dataloader, device, args.batch_size, args.model1)
    print(f"Number of wrong predictions from {args.model1}: {len(wrong_indices)} for an accuracy of {(len(train_dataset) - len(wrong_indices)) / len(train_dataset) * 100:.2f}%\n")
    
        # Load first model
    if DATASET == 'IMAGENET':
        model = get_model(args.model2, weights=imagenet_models[args.model2]).to(device)
    else:
        model = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{args.model2}', pretrained=True).to(device)

    if args.method == "2":
        # Get correct predictions from second model from proper dataset
        correct_indices = gather_correct_preds(model, train_dataloader, device, args.batch_size, args.model2)
        indices = list(set(wrong_indices + correct_indices)) # Keep each index once to be sure
        # Create a subset from the incorrect predictions of the first model and correct predictions from the second model
        print(f"Creating subset from {args.model1} incorrect predictions and {args.model2} correct predictions.")
    else:        
        indices = wrong_indices
        # Create a subset from the incorrect predictions of the first model
        print(f"Creating subset from {args.model1} incorrect predictions.")


    subset = Subset(dataset=train_dataset, indices=indices)
    subset_dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Test accuracy of second model on subset before fine tuning 
    correct = measure_accuracy(model, subset_dataloader, device, args.model2)
    print(f"Accuracy: {correct / len(subset) * 100:.2f}%")

    # Create instances of functions and classes for training
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0)
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(model=model, 
                    optimizer=optimizer, 
                    train_dataloader=subset_dataloader,
                    # validation_dataloader = subset_dataloader,
                    loss_fn=loss_fn,
                    epochs=args.epochs_model2,
                    device=device)

    # lol?
    while True:
        ans = input("Run a fine tuning step? y/n: ")
        if ans == "y":
            print(f"Fine tuning {args.model2} on subset.")
            trainer.train_model()
            correct = measure_accuracy(model, subset_dataloader, device, args.model2)
            print(f"Accuracy: {correct / len(subset) * 100:.2f}%")
        elif ans == "n":
            break

    # Save dictionary of second fine tuned model
    trainer.save_model_weights(filepath=f'./{args.model2}-FT({args.model1}).pth', append_accuracy=False)
    print(f"\nSaved model weigts as '{args.model2}-FT({args.model1}).pth'")


    # If fine tune first model option is set
    if args.fine_tune_first:
        # Gather wrong predictions from second fine tuned model from proper dataset
        wrong_indices = gather_wrong_preds(model, train_dataloader, device, args.batch_size, f'{args.model2}-FT({args.model1})')
        print(f"Number of wrong predictions from {args.model2}-FT({args.model1}): {len(wrong_indices)} for an accuracy of {(len(train_dataset) - len(wrong_indices)) / len(train_dataset) * 100:.2f}%\n")

        # Load first model
        if DATASET == 'IMAGENET':
            model = get_model(args.model1, weights=imagenet_models[args.model1]).to(device)
        else:
            model = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{args.model1}', pretrained=True).to(device)
            
        if args.method == "2":
            # Gather correct predictions from first model from proper dataset
            correct_indices = gather_correct_preds(model, train_dataloader, device, args.batch_size, args.model1)
            indices = list(set(wrong_indices + correct_indices)) # Keep each index once to be sure
            # Create a subset from the incorrect predictions of the first model and correct predictions from the second model
            print(f"Creating subset from {args.model2}-FT({args.model1}) incorrect predictions and {args.model1} correct predictions.")
        else:        
            indices = wrong_indices
            # Create a subset from the incorrect predictions of the first model
            print(f"Creating subset from {args.model2}-FT({args.model1}) incorrect predictions.")

        '''
        NOTE   
        subset = wrong predictions from second fine tuned model + correct predictions from first model
        The higher accuracy of the second model, is due to the fact that the wrong predictions from the first fine tuned model are very few
        and as such the correct predictions from the second model plus the few incorrect from the first model leads to > 90% acc.
        '''

        # Create a subset of the wrong predictions of the second fine tuned model
        subset = Subset(dataset=train_dataset, indices=indices)
        subset_dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        # Test accuracy of first model on subset before fine tuning 
        correct = measure_accuracy(model, subset_dataloader, device, args.model1)
        print(f"Accuracy: {correct / len(subset) * 100:.2f}%")

        cuda.empty_cache()

        # Create instances of classes and functions for training first model
        optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0)
        loss_fn = CrossEntropyLoss()
        trainer = Trainer(model=model, 
                        optimizer=optimizer, 
                        train_dataloader=subset_dataloader,
                        #   validation_dataloader = valloader,
                        loss_fn=loss_fn,
                        epochs=1,
                        device=device)
        
        while True:
            ans = input("Run a fine tuning step? y/n: ")
            if ans == "y":
                print(f"Fine tuning {args.model1} on subset.")
                trainer.train_model()
                correct = measure_accuracy(model, subset_dataloader, device, args.model1)
                print(f"Accuracy: {correct / len(subset) * 100:.2f}%")
            elif ans == "n":
                break

        # Save weights of second model
        save(model.state_dict(), f'./{args.model1}-FT({args.model2}-FT({args.model1})).pth')
        print(f"\nSaved model weigts as '{args.model1}-FT({args.model2}-FT({args.model1})).pth'")


if __name__ == '__main__':
    main()

