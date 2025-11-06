import numpy as np
import csv
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
from captum.attr import IntegratedGradients

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class DenoisingDiffusion(nn.Module):
    def __init__(self, input_dim, num_timesteps=1000):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.input_dim = input_dim
        self.beta = self._linear_beta_schedule(num_timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, input_dim)

    def _linear_beta_schedule(self, timesteps, beta_start=0.0001, beta_end=0.02):
        return torch.linspace(beta_start, beta_end, timesteps)

    def forward(self, xt, t):
        x = torch.relu(self.fc1(xt))
        x = torch.relu(self.fc2(x))
        x = x + self.fc3(x) * t.view(-1, 1)
        x = self.fc4(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads=4, num_layers=1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 64)
        encoder_layers = nn.TransformerEncoderLayer(d_model=64, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        return self.fc(x[:, -1, :]), x[:, -1, :]

class DeepSurvLoss(nn.Module):
    def forward(self, predictions, y_time, y_event):
        eps = 1e-7
        predictions = predictions.reshape(-1)
        y_time = y_time.reshape(-1)
        y_event = y_event.reshape(-1)
        order = torch.argsort(y_time, descending=True)
        pred_sorted = predictions[order]
        event_sorted = y_event[order]
        exp_pred = torch.exp(pred_sorted)
        log_cumsum = torch.log(torch.cumsum(exp_pred, dim=0) + eps)
        loss = -torch.sum((pred_sorted - log_cumsum) * event_sorted) / (torch.sum(event_sorted) + eps)
        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, features1, features2):
        batch_size = features1.shape[0]
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        labels = torch.arange(batch_size).to(features1.device)
        similarity_matrix = torch.matmul(features1, features2.T) / self.temperature
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

class CombinedModel(nn.Module):
    def __init__(self, input_dim1, input_dim2, input_dim3, hidden_dim):
        super().__init__()
        self.diffusion_mod1 = DenoisingDiffusion(input_dim1)
        self.diffusion_mod2 = DenoisingDiffusion(input_dim2)
        self.diffusion_mod3 = DenoisingDiffusion(input_dim3)
        self.align_mod1 = nn.Linear(input_dim1, hidden_dim)
        self.align_mod2 = nn.Linear(input_dim2, hidden_dim)
        self.align_mod3 = nn.Linear(input_dim3, hidden_dim)
        self.transformer_mod1 = TransformerModel(hidden_dim)
        self.transformer_mod2 = TransformerModel(hidden_dim)
        self.transformer_mod3 = TransformerModel(hidden_dim)
        self.weight_mod1 = nn.Parameter(torch.tensor(0.33))
        self.weight_mod2 = nn.Parameter(torch.tensor(0.33))
        self.weight_mod3 = nn.Parameter(torch.tensor(0.34))
        self.final_fc = nn.Linear(3, 1)

    def forward(self, mod1, mod2, mod3, t):
        mod1_denoised = self.diffusion_mod1(mod1, t)
        mod1_aligned = self.align_mod1(mod1_denoised)
        out1, feat1 = self.transformer_mod1(mod1_aligned)

        mod2_denoised = self.diffusion_mod2(mod2, t)
        mod2_aligned = self.align_mod2(mod2_denoised)
        out2, feat2 = self.transformer_mod2(mod2_aligned)

        mod3_denoised = self.diffusion_mod3(mod3, t)
        mod3_aligned = self.align_mod3(mod3_denoised)
        out3, feat3 = self.transformer_mod3(mod3_aligned)

        final_output = torch.cat((
            self.weight_mod1 * out1,
            self.weight_mod2 * out2,
            self.weight_mod3 * out3
        ), dim=1)

        return self.final_fc(final_output), feat1, feat2, feat3

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

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


grid_params = {
    "seeds": [8],
    "learning_rates": [1e-2],
    "batch_sizes": [1024],
    "hidden_dims": [500],
    "alpha_contrastive": [0.8]
}

overall_results_file = 'grid_search_results.csv'
with open(overall_results_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['seed', 'learning_rate', 'batch_size', 'hidden_dim', 'alpha_contrastive', 'max_mean_ci'])


for seed in grid_params["seeds"]:
    for lr in grid_params["learning_rates"]:
        for batch_size in grid_params["batch_sizes"]:
            for hidden_dim in grid_params["hidden_dims"]:
                for alpha_contrastive in grid_params["alpha_contrastive"]:

                    file_names = [
                        ("mrna", "data/cos/test_mrna.csv"),
                        ("mirna", "data/sp/test_mi.csv"),
                        ("meth", "data/pea/test_log_meth.csv")
                    ]
                    datasets = []
                    time_tensors = []
                    event_tensors = []
                    feature_names_all = []
                    for prefix, file_name in file_names:
                        df = pd.read_csv(file_name, header=0)
                        survival_time = pd.to_numeric(df.iloc[0, 1:], errors='coerce').values
                        survival_status = pd.to_numeric(df.iloc[1, 1:], errors='coerce').values

                        feature_names = df.iloc[2:, 0].tolist()
                        feature_names_prefixed = [f"{prefix}_{name}" for name in feature_names]
                        feature_names_all.extend(feature_names_prefixed)

                        gene_data = df.iloc[2:, 1:].values
                        mod = torch.tensor(gene_data, dtype=torch.float32).T
                        time_tensor = torch.tensor(survival_time, dtype=torch.float32)
                        event_tensor = torch.tensor(survival_status, dtype=torch.float32)
                        datasets.append(mod)
                        time_tensors.append(time_tensor)
                        event_tensors.append(event_tensor)

                    mod1, mod2, mod3 = datasets
                    y_time = torch.cat(time_tensors)[:mod1.shape[0]]
                    y_event = torch.cat(event_tensors)[:mod1.shape[0]]
                    num_samples = mod1.shape[0]
                    input_dim1, input_dim2, input_dim3 = mod1.shape[1], mod2.shape[1], mod3.shape[1]

                    print(f"\n=====  seed={seed}, lr={lr}, batch_size={batch_size}, hidden_dim={hidden_dim}, alpha_contrastive={alpha_contrastive} =====")
                    params = {
                        "seed": seed,
                        "num_layer": 1,
                        "hidden_dim": hidden_dim,
                        "num_epochs": 10,
                        "learning_rate": lr,
                        "batch_size": batch_size,
                        "alpha_contrastive": alpha_contrastive
                    }
                    set_seed(params["seed"])

                    t = torch.randint(0, 100, (num_samples,), generator=torch.Generator().manual_seed(seed))
                    output_dir = f"output_layer{params['num_layer']}_dim{params['hidden_dim']}_ep{params['num_epochs']}_lr{params['learning_rate']}_bs{params['batch_size']}_alpha{params['alpha_contrastive']}_seed{params['seed']}"
                    os.makedirs(output_dir, exist_ok=True)


                    n_splits = 5
                    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
                    folds_cindex = []
                    folds_ig = []

                    original_mod1 = mod1.clone().to(device)
                    original_mod2 = mod2.clone().to(device)
                    original_mod3 = mod3.clone().to(device)

                    for fold, (train_index, val_index) in enumerate(kf.split(original_mod1)):
                        print(f"Fold {fold + 1}/{n_splits}")
                        mod1_train, mod1_val = original_mod1[train_index], original_mod1[val_index]
                        mod2_train, mod2_val = original_mod2[train_index], original_mod2[val_index]
                        mod3_train, mod3_val = original_mod3[train_index], original_mod3[val_index]
                        y_time_train, y_time_val = y_time[train_index], y_time[val_index]
                        y_event_train, y_event_val = y_event[train_index], y_event[val_index]

                        t_train = torch.randint(0, 100, (len(train_index),), generator=torch.Generator().manual_seed(seed))
                        t_val = torch.randint(0, 100, (len(val_index),), generator=torch.Generator().manual_seed(seed))

                        train_dataset = TensorDataset(mod1_train, mod2_train, mod3_train, y_time_train, y_event_train, t_train)
                        val_dataset = TensorDataset(mod1_val, mod2_val, mod3_val, y_time_val, y_event_val, t_val)

                        train_loader = DataLoader(
                            train_dataset,
                            batch_size=params['batch_size'],
                            shuffle=True,
                            worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
                        )
                        val_loader = DataLoader(
                            val_dataset,
                            batch_size=params['batch_size'],
                            shuffle=False
                        )

                        set_seed(seed)
                        model = CombinedModel(input_dim1, input_dim2, input_dim3, hidden_dim=params['hidden_dim']).to(device)
                        initialize_weights(model)

                        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
                        criterion = DeepSurvLoss()
                        contrastive_criterion = ContrastiveLoss(temperature=0.5)

                        fold_cindex = []


                        for epoch in range(params["num_epochs"]):
                            set_seed(seed + epoch)
                            model.train()
                            total_loss = 0
                            for mod1_b, mod2_b, mod3_b, batch_y_time, batch_y_event, batch_t in train_loader:
                                mod1_b, mod2_b, mod3_b = mod1_b.to(device), mod2_b.to(device), mod3_b.to(device)
                                batch_y_time, batch_y_event, batch_t = batch_y_time.to(device), batch_y_event.to(device), batch_t.to(device)

                                optimizer.zero_grad()
                                outputs, feat1, feat2, feat3 = model(mod1_b, mod2_b, mod3_b, batch_t)
                                loss_surv = criterion(outputs, batch_y_time.float(), batch_y_event.float())
                                loss_contrastive = contrastive_criterion(feat1, feat2) + \
                                                   contrastive_criterion(feat1, feat3) + \
                                                   contrastive_criterion(feat2, feat3)
                                loss = loss_surv + alpha_contrastive * loss_contrastive
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                                optimizer.step()
                                total_loss += loss.item()


                            model.eval()
                            all_predictions, all_true_time, all_true_event = [], [], []
                            with torch.no_grad():
                                for mod1_b, mod2_b, mod3_b, batch_y_time, batch_y_event, batch_t in val_loader:
                                    mod1_b, mod2_b, mod3_b, batch_t = mod1_b.to(device), mod2_b.to(device), mod3_b.to(device), batch_t.to(device)
                                    outputs, _, _, _ = model(mod1_b, mod2_b, mod3_b, batch_t)
                                    prediction = outputs.cpu().numpy().reshape(-1)
                                    all_predictions.append(prediction)
                                    all_true_time.append(batch_y_time.numpy().reshape(-1))
                                    all_true_event.append(batch_y_event.numpy().reshape(-1))

                            all_predictions = np.concatenate(all_predictions, axis=0)
                            all_true_time = np.concatenate(all_true_time, axis=0)
                            all_true_event = np.concatenate(all_true_event, axis=0)

                            ci_value = compute_ci(torch.tensor(all_predictions), torch.tensor(all_true_time),
                                                  torch.tensor(all_true_event))
                            fold_cindex.append(ci_value)
                            print(f"Epoch {epoch + 1} C-index: {ci_value:.4f}")

                            pred_dir = os.path.join(output_dir, "predictions")
                            save_predictions(fold, epoch, all_true_time, all_true_event, all_predictions, pred_dir)

                        folds_cindex.append(fold_cindex)


                        class WrapperModel(torch.nn.Module):
                            def __init__(self, model):
                                super().__init__()
                                self.model = model
                            def forward(self, x):
                                batch_size = x.shape[0]
                                dim1 = self.model.diffusion_mod1.input_dim
                                dim2 = self.model.diffusion_mod2.input_dim
                                dim3 = self.model.diffusion_mod3.input_dim
                                mod1, mod2, mod3 = x[:, :dim1], x[:, dim1:dim1+dim2], x[:, dim1+dim2:]
                                t_tensor = torch.randint(0, 100, (batch_size,), device=x.device)
                                out, _, _, _ = self.model(mod1, mod2, mod3, t_tensor)
                                return out

                        val_input = torch.cat((mod1_val, mod2_val, mod3_val), dim=1).to(device)
                        baseline = torch.zeros_like(val_input)
                        wrapper_model = WrapperModel(model).to(device)
                        ig = IntegratedGradients(wrapper_model)
                        attr_ig = ig.attribute(val_input, baselines=baseline, n_steps=50).detach().cpu().numpy()
                        folds_ig.append(attr_ig)

                    df_ig = pd.DataFrame({"feature": feature_names_all})

                    for fold_idx, ig_values in enumerate(folds_ig):
                        ig_values = np.array(ig_values)
                        if ig_values.ndim == 2:
                            ig_values = ig_values.mean(axis=0)

                        max_abs = np.max(np.abs(ig_values))
                        if max_abs > 0:
                            ig_values_norm = ig_values / max_abs
                        else:
                            ig_values_norm = np.zeros_like(ig_values)

                        df_ig[f"fold{fold_idx}"] = ig_values_norm.flatten()

                    df_ig["mean"] = df_ig[[f"fold{i}" for i in range(len(folds_ig))]].mean(axis=1)

                    df_ig.to_csv(
                        f"ig_all_norm_signed_seed{seed}_lr{lr}_bs{batch_size}_hidden{hidden_dim}_alpha{alpha_contrastive}.csv",
                        index=False
                    )
                    df_ig["mean"] = df_ig[[f"fold{i}" for i in range(len(folds_ig))]].mean(axis=1)

                    df_ig.to_csv(
                        f"ig_all_seed{seed}_lr{lr}_bs{batch_size}_hidden{hidden_dim}_alpha{alpha_contrastive}.csv",
                        index=False
                    )

                    cv_results_file = os.path.join(output_dir, 'ci_results.csv')
                    with open(cv_results_file, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        header = [f'epoch_{i + 1}' for i in range(params["num_epochs"])]
                        writer.writerow(header)
                        for cindex_values in folds_cindex:
                            writer.writerow(cindex_values)
                        epoch_means = np.mean(folds_cindex, axis=0)
                        writer.writerow(epoch_means)
                        max_mean = np.max(epoch_means)
                        writer.writerow([max_mean])

                    with open(overall_results_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([seed, lr, batch_size, hidden_dim, alpha_contrastive, max_mean])

                    print(f"finished: seed={seed}, lr={lr}, batch_size={batch_size}, hidden_dim={hidden_dim}, alpha_contrastive={alpha_contrastive}, best C-index={max_mean:.4f}")

print("ALL FINISHED")
