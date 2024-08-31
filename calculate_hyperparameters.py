from torch import cuda, inference_mode, argmax, tensor, load, manual_seed, backends, hub
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.models import get_model

from utils.models_lists import imagenet_models
from utils.heuristics import heuristic_search_process
from utils.reporting import run_methodology_with_postcheck_and_calculate_classification_report2, run_methodology_and_calculate_classification_report2
from utils.datasets import ImageNetVal
from utils.score_fns import max_probability, difference, entropy

from sklearn.metrics import classification_report
from argparse import ArgumentParser
from pandas import read_csv, DataFrame
from numpy import linspace
from tqdm.auto import tqdm
from multiprocessing.pool import Pool

def main():
    parser = ArgumentParser()
    parser.add_argument("-m1", "--model1", help="The first model, required. This parameter will set which dataset to use (CIFAR10 or ImageNet)", required=True)   
    parser.add_argument("-m2", "--model2", help="The second model.", default=None, required=False)
    parser.add_argument("-f", "--use-fine-tuned", help="If set, it will use the saved model weights from fine tuned models otherwise it will use the standard pretrained weights from PyTorch.", required=False, default=False, action="store_true")
    parser.add_argument("-d", "--data-path", help="The path where the datasets (CIFAR-10 or custom ImageNet) are.")
    parser.add_argument("-b", "--batch-size", help="The size of each batch", default=32, type=int)
    parser.add_argument("-t", "--threshold-points", help="How many threshold points between 0 and 1 should the hypermarameter search examine. Default is 4000", default=4000, type=int)
    parser.add_argument("-w", "--worker-threads", help="How many worker threads should the hyperparamter search use. Default is 16", default=16, type=int)
    parser.add_argument("-s", "--seed", help="The seed to be used in all pytorch classes, for reproducibility.", default=None)    
    args = parser.parse_args()

    # Constants
    DATA_FILEPATH = args.data_path
    BATCH_SIZE = args.batch_size
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


    if DATASET == 'IMAGENET':
        print("Using ImageNet dataset")
        n_classes = 1000
        entropy_norm_factor = entropy(tensor([1/n_classes for _ in range(n_classes)]).unsqueeze(dim=0))
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
        valset = ImageNetVal(image_path=f'{DATA_FILEPATH}/ImageNet/val_resized_40K', transform=transform, random_seed=42)
        valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        testset = ImageNetVal(image_path=f'{DATA_FILEPATH}/ImageNet/val_resized_10K', transform=transform, random_seed=42)
        testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    else:
        print("Using CIFAR10 dataset")
        n_classes = 10
        entropy_norm_factor = entropy(tensor([1/n_classes for _ in range(n_classes)]).unsqueeze(dim=0))
        transform = Compose([
                ToTensor(),
                Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201)) # CIFAR10
        ])
        valset = CIFAR10(root=DATA_FILEPATH, download=True, train=True, transform=transform)
        valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        testset = CIFAR10(root=DATA_FILEPATH, download=True, train=False, transform=transform)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)


    models = [args.model1, args.model2]
    modelsft = [f"{models[0]}-FT({models[1]}-FT({models[0]}))", f"{models[1]}-FT({models[0]})"]

    val_true = []
    test_true = []

    for i, (model_name, model_name_ft) in enumerate(zip(models, modelsft)):    
        print(f"Geting predictions from {model_name_ft} validation set.")

        if DATASET == 'IMAGENET':
            model = get_model(model_name, weights=imagenet_models[model_name]).to(device)
        else:
            model = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{model_name}', pretrained=True).to(device)
        
        if args.use_fine_tuned:
            model.load_state_dict(load(f"./{model_name_ft}.pth"))

        model.eval()

        report = []
        with inference_mode():
            for X, y in tqdm(valloader):                
                if i == 0: val_true += y.tolist()                    
                X = X.to(device)
                preds = model(X)
                for pred in preds:
                    report.append({
                        'classification': argmax(pred).item(),
                        'max_prob' : max_probability(pred.unsqueeze(dim=0)),
                        'difference' : difference(pred.unsqueeze(dim=0)),
                        'entropy' : entropy(pred.unsqueeze(dim=0)) / entropy_norm_factor
                    })

        df = DataFrame(report)
        df.to_csv(f'./{DATASET}-valset-{model_name_ft}.csv', index=False)

        print(f"Geting predictions from {model_name_ft} test set.")
        report = []
        with inference_mode():
            for X, y in tqdm(testloader):
                if i == 0: test_true += y.tolist()                    
                X = X.to(device)
                preds = model(X)
                for pred in preds:
                    report.append({
                        'classification': argmax(pred).item(),
                        'max_prob' : max_probability(pred.unsqueeze(dim=0)),
                        'difference' : difference(pred.unsqueeze(dim=0)),
                        'entropy' : entropy(pred.unsqueeze(dim=0)) / entropy_norm_factor
                    })

        df = DataFrame(report)
        df.to_csv(f'./{DATASET}-testset-{model_name_ft}.csv', index=False)

    df_val_preds_a = read_csv(f'./{DATASET}-valset-{modelsft[0]}.csv') 
    df_test_preds_a = read_csv(f'./{DATASET}-testset-{modelsft[0]}.csv')
    df_val_preds_b = read_csv(f'./{DATASET}-valset-{modelsft[1]}.csv') 
    df_test_preds_b = read_csv(f'./{DATASET}-testset-{modelsft[1]}.csv')
    df_val_true = DataFrame(val_true, columns=['true'])
    df_test_true = DataFrame(test_true, columns=['true'])


    threshold_params = linspace(0, 1, args.threshold_points)
    step = args.threshold_points // args.worker_threads

    maxp_splits = []
    diff_splits = []
    entropy_splits = []
    maxp_rev_splits = []
    diff_rev_splits = []
    entropy_rev_splits = []

    for i in range(args.worker_threads):
        maxp_splits.append((df_val_true['true'], df_val_preds_a['classification'], df_val_preds_b['classification'], df_val_preds_a['max_prob'], df_val_preds_b['max_prob'], threshold_params[i * step : (i+1) * step], False))
        diff_splits.append((df_val_true['true'], df_val_preds_a['classification'], df_val_preds_b['classification'], df_val_preds_a['difference'], df_val_preds_b['difference'], threshold_params[i * step : (i+1) * step], False))
        entropy_splits.append((df_val_true['true'], df_val_preds_a['classification'], df_val_preds_b['classification'], df_val_preds_a['entropy'], df_val_preds_b['entropy'], threshold_params[i * step : (i+1) * step], True))
        maxp_rev_splits.append((df_val_true['true'], df_val_preds_b['classification'], df_val_preds_a['classification'], df_val_preds_b['max_prob'], df_val_preds_a['max_prob'], threshold_params[i * step : (i+1) * step], False))
        diff_rev_splits.append((df_val_true['true'], df_val_preds_b['classification'], df_val_preds_a['classification'], df_val_preds_b['difference'], df_val_preds_a['difference'], threshold_params[i * step : (i+1) * step], False))
        entropy_rev_splits.append((df_val_true['true'], df_val_preds_b['classification'], df_val_preds_a['classification'], df_val_preds_b['entropy'], df_val_preds_a['entropy'], threshold_params[i * step : (i+1) * step], True))

    maxp_results = []
    maxp_ps_results = []
    diff_results = []
    diff_ps_results = []
    entropy_results = []
    entropy_ps_results = []
    print("Running λ hyper-parameter search please wait...")
    # create a thread pool
    with Pool(args.worker_threads) as pool:
        print("Running Max Probability")
        for ret in pool.starmap(heuristic_search_process, maxp_splits):
            maxp_results += ret[0]
            maxp_ps_results += ret[1]

        print("Running Difference")
        for ret in pool.starmap(heuristic_search_process, diff_splits):
            diff_results += ret[0]
            diff_ps_results += ret[1]

        print("Running Entropy")
        for ret in pool.starmap(heuristic_search_process, entropy_splits):
            entropy_results += ret[0]
            entropy_ps_results += ret[1]

    df_max_p = DataFrame(maxp_results)
    df_diff = DataFrame(diff_results)
    df_entropy = DataFrame(entropy_results).iloc[::-1].reset_index(drop=True) # Reverse order for entropy
    df_max_p_ps = DataFrame(maxp_ps_results)
    df_diff_ps = DataFrame(diff_ps_results)
    df_entropy_ps = DataFrame(entropy_ps_results).iloc[::-1].reset_index(drop=True) # Reverse order for entropy

    # Normal
    max_p_param = df_max_p.iloc[df_max_p['accuracy'].idxmax()].iat[0]
    diff_param = df_diff.iloc[df_diff['accuracy'].idxmax()].iat[0]
    entropy_param = df_entropy.iloc[df_entropy['accuracy'].idxmax()].iat[0]
    max_p_ps_param = df_max_p_ps.iloc[df_max_p_ps['accuracy'].idxmax()].iat[0]
    diff_ps_param = df_diff_ps.iloc[df_diff_ps['accuracy'].idxmax()].iat[0]
    entropy_ps_param = df_entropy_ps.iloc[df_entropy_ps['accuracy'].idxmax()].iat[0]

    results = []

    result, thr, usage = run_methodology_and_calculate_classification_report2(df_test_true['true'], 
                                                        df_test_preds_a['classification'], 
                                                        df_test_preds_b['classification'], 
                                                        df_test_preds_a['max_prob'], 
                                                        df_test_preds_b['max_prob'], 
                                                        max_p_param,
                                                        models[0],
                                                        models[1])

    results.append({
        'score_fn': 'maxp',
        'threshold' : thr,
        'second_model_usage': usage,
        'accuracy':  result['accuracy'],
        'precision': result['macro avg']['precision'],
        'recall': result['macro avg']['recall'],
        'f1': result['macro avg']['f1-score']
    })

    result, thr, usage = run_methodology_and_calculate_classification_report2(df_test_true['true'], 
                                                        df_test_preds_a['classification'], 
                                                        df_test_preds_b['classification'], 
                                                        df_test_preds_a['difference'], 
                                                        df_test_preds_b['difference'], 
                                                        diff_param,
                                                        models[0],
                                                        models[1])

    results.append({
        'score_fn': 'difference',
        'threshold' : thr,
        'second_model_usage': usage,
        'accuracy':  result['accuracy'],
        'precision': result['macro avg']['precision'],
        'recall': result['macro avg']['recall'],
        'f1': result['macro avg']['f1-score']
    })

    result, thr, usage = run_methodology_and_calculate_classification_report2(df_test_true['true'], 
                                                        df_test_preds_a['classification'], 
                                                        df_test_preds_b['classification'], 
                                                        df_test_preds_a['entropy'], 
                                                        df_test_preds_b['entropy'], 
                                                        entropy_param, 
                                                        models[0],
                                                        models[1])

    results.append({
        'score_fn': 'entropy',
        'threshold' : thr,
        'second_model_usage': usage,
        'accuracy':  result['accuracy'],
        'precision': result['macro avg']['precision'],
        'recall': result['macro avg']['recall'],
        'f1': result['macro avg']['f1-score']
    })

    result, thr, usage = run_methodology_with_postcheck_and_calculate_classification_report2(df_test_true['true'], 
                                                                                df_test_preds_a['classification'], 
                                                                                df_test_preds_b['classification'], 
                                                                                df_test_preds_a['max_prob'], 
                                                                                df_test_preds_b['max_prob'], 
                                                                                max_p_ps_param,
                                                                                models[0],
                                                                                models[1])

    results.append({
        'score_fn': 'maxp_ps',
        'threshold' : thr,
        'second_model_usage': usage,
        'accuracy':  result['accuracy'],
        'precision': result['macro avg']['precision'],
        'recall': result['macro avg']['recall'],
        'f1': result['macro avg']['f1-score']
    })

    result, thr, usage = run_methodology_with_postcheck_and_calculate_classification_report2(df_test_true['true'], 
                                                                                df_test_preds_a['classification'], 
                                                                                df_test_preds_b['classification'], 
                                                                                df_test_preds_a['difference'], 
                                                                                df_test_preds_b['difference'], 
                                                                                diff_ps_param,
                                                                                models[0],
                                                                                models[1])

    results.append({
        'score_fn': 'diff_ps',
        'threshold' : thr,
        'second_model_usage': usage,
        'accuracy':  result['accuracy'],
        'precision': result['macro avg']['precision'],
        'recall': result['macro avg']['recall'],
        'f1': result['macro avg']['f1-score']
    })

    result, thr, usage = run_methodology_with_postcheck_and_calculate_classification_report2(df_test_true['true'], 
                                                                                df_test_preds_a['classification'], 
                                                                                df_test_preds_b['classification'], 
                                                                                df_test_preds_a['entropy'], 
                                                                                df_test_preds_b['entropy'], 
                                                                                entropy_ps_param, 
                                                                                models[0],
                                                                                models[1])

    results.append({
        'score_fn': 'entropy_ps',
        'threshold' : thr,
        'second_model_usage': usage,
        'accuracy':  result['accuracy'],
        'precision': result['macro avg']['precision'],
        'recall': result['macro avg']['recall'],
        'f1': result['macro avg']['f1-score']
    })


    best = max(results, key=lambda x: x['accuracy'])

    singlea = classification_report(df_test_true['true'], df_test_preds_a['classification'], digits = 4, zero_division=0, output_dict = True)
    singleb = classification_report(df_test_true['true'], df_test_preds_b['classification'], digits = 4, zero_division=0, output_dict = True)

    print(f"Best Performance parameters")
    print(f"A: {modelsft[0]}: {singlea['accuracy']:.4f} | B: {modelsft[1]}: {singleb['accuracy']:.4f}")
    print(f"Score function: {best['score_fn']} | Threshold: {best['threshold']:.4f} | Second model usage: {best['second_model_usage'] * 100:.2f}%")
    print(f"Accuracy: {best['accuracy']:.4f} | Precision: {best['precision']:.4f} | Recall: {best['recall']:.4f} | F1: {best['f1']:.4f}")


if __name__ == "__main__":
    main()