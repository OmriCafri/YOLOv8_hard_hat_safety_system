import ultralytics
import torch
import optuna
import yaml
import joblib
import os

with open('../configurations.yaml', 'r') as f:
    conf = list(yaml.load_all(f, Loader=yaml.SafeLoader))[0]

# Creating a template for saving the training results
template = r'{}_optimizer_{}_epochs_{}_patience'
# training results directory
results_dir = r'runs/detect/'

# For this optimization we will use optuna like grid-search
# we can remove sampler='GridSampler' if we have large number of combinations and can't run all of them


def _train_new_model(optimizer, epochs, patience):
    """
    Function for training new YOLO instance

    Parameters
    ----------
    optimizer: str
        Optimizer name
    epochs: int
        The number of epochs
    patience: int
        The max number of epochs without improvement for early stopping

    Returns
    -------
    str
        The best model path
    """
    # Reading pretrained model (on COCO dataset)
    model = ultralytics.YOLO(r'coco_pretrained_model/yolov8n.pt')
    # Training on custom data
    model.train(optimizer=optimizer,
                epochs=epochs,
                patience=patience,
                data=conf['data_yaml_path'],
                imgsz=416,
                pretrained=True,
                workers=4,
                device=0,
                seed=42,
                name=template.format(optimizer,
                                     epochs,
                                     patience))

    # Getting the best model for this option
    return model.model.pt_path


def objective(trail):
    """ Objective function for running with optuna"""
    # Suggesting params using optuna
    optimizer = trail.suggest_categorical('optimizer', ['SGD', 'Adam', 'RMSProp'])
    # Setting the number of epochs for this trail
    epochs = 300
    # Setting patience
    patience = trail.suggest_categorical('patience', [10, 25, 50])

    # if we didn't run this iteration already
    if not os.path.exists(results_dir + template.format(optimizer, epochs, patience)):
        best_model_path = _train_new_model(optimizer, epochs, patience)
    else:
        best_model_path = results_dir + template.format(optimizer, epochs, patience) + '/weights/best.pt'

    # Creating an instance based on the best model weights
    model = ultralytics.YOLO(best_model_path)
    # Validating the model
    model.val(split='val',
              name=template.format(optimizer,
                                   epochs,
                                   patience) + "_val")
    # returning the fitness measure
    # (fitness = 0.9 * mAP50-95 + 0.1 * mAP50)
    return model.metrics.fitness


def create_study(n_trails, sampler, saving_path=None):
    """
    Function for creating an optuna study and running it.

    Parameters
    ----------
    n_trails: int
        The number of trails
    sampler: optuna sampler
        Sampler to use
    saving_path: str (default None)
        Path for saving study results
    """
    # Creating a study to maximize fitness
    study = optuna.create_study(direction='maximize', sampler=sampler)
    # Creating a function to optimize based on objective function
    func_optimize = lambda trail: objective(trail)
    # Starting optimization
    study.optimize(func_optimize, n_trials=n_trails)
    # Printing best params
    print('Best params:')
    print(study.best_params)
    # Printing best fitness
    print('Best fitness')
    print(study.best_value)
    # saving study
    if saving_path is not None:
        joblib.dump(study, saving_path)


if __name__ == '__main__':
    # Checking if a gpu is available
    if not torch.cuda.is_available():
        raise Exception('GPU is not available')
    print(torch.cuda.get_device_name(0))

    # creating sampler
    search_space = {
        'optimizer': ['SGD', 'Adam', 'RMSProp'],
        'patience': [10, 25, 50]
    }
    sampler = optuna.samplers.GridSampler(search_space)

    # Running study
    create_study(9, sampler, saving_path="optuna_study_results.pkl", )
