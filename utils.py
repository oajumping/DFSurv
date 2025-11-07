import os
import random
import numpy as np
import torch
import pandas as pd
from lifelines.utils import concordance_index

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

def compute_ci(outputs, y_time, y_event):
    predictions = outputs.detach().cpu().numpy().flatten()
    event_times = y_time.cpu().numpy().flatten()
    event_observed = y_event.cpu().numpy().flatten()
    return concordance_index(event_times, predictions, event_observed)

def save_predictions(fold, epoch, true_time, true_event, predictions, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(true_time, torch.Tensor):
        true_time = true_time.detach().cpu().numpy()
    if isinstance(true_event, torch.Tensor):
        true_event = true_event.detach().cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    df = pd.DataFrame({
        'fold': fold + 1,
        'epoch': epoch + 1,
        'Sample_Index': range(len(predictions)),
        'True_Time': true_time.squeeze(),
        'True_Event': true_event.squeeze(),
        'Predicted': predictions.squeeze()
    })
    file_path = os.path.join(output_dir, f'predictions_epoch_{epoch + 1}.csv')
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, mode='w', header=True, index=False)
