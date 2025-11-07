import os
import csv
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from captum.attr import IntegratedGradients

from model_components import CombinedModel, DeepSurvLoss, ContrastiveLoss
from utils import set_seed, initialize_weights, compute_ci, save_predictions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- grid_params ----------
grid_params = {
    "seeds": [8],
    "learning_rates": [1e-2],
    "batch_sizes": [1024],
    "hidden_dims": [500],
    "alpha_contrastive": [0.5]
}

overall_results_file = 'grid_search_results.csv'
with open(overall_results_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['seed', 'learning_rate', 'batch_size', 'hidden_dim', 'alpha_contrastive', 'max_mean_ci'])

# ---------- training ----------
for seed in grid_params["seeds"]:
    for lr in grid_params["learning_rates"]:
        for batch_size in grid_params["batch_sizes"]:
            for hidden_dim in grid_params["hidden_dims"]:
                for alpha_contrastive in grid_params["alpha_contrastive"]:
                    file_names = [
                        ("mrna", "sample/cos/test_mrna.csv"),
                        ("mirna", "sample/sp/test_mi.csv"),
                        ("meth", "sample/pea/test_meth.csv")
                    ]
                    datasets, time_tensors, event_tensors, feature_names_all = [], [], [], []

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

                    params = {"seed": seed, "num_layer": 1, "hidden_dim": hidden_dim,
                              "num_epochs": 10, "learning_rate": lr, "batch_size": batch_size,
                              "alpha_contrastive": alpha_contrastive}
                    set_seed(seed)

                    t = torch.randint(0, 100, (num_samples,), generator=torch.Generator().manual_seed(seed))
                    output_dir = f"output_layer{params['num_layer']}_dim{params['hidden_dim']}_ep{params['num_epochs']}_lr{params['learning_rate']}_bs{params['batch_size']}_alpha{params['alpha_contrastive']}_seed{params['seed']}"
                    os.makedirs(output_dir, exist_ok=True)

                    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
                    folds_cindex, folds_ig = [], []

                    original_mod1, original_mod2, original_mod3 = mod1.to(device), mod2.to(device), mod3.to(device)

                    for fold, (train_index, val_index) in enumerate(kf.split(original_mod1)):
                        mod1_train, mod1_val = original_mod1[train_index], original_mod1[val_index]
                        mod2_train, mod2_val = original_mod2[train_index], original_mod2[val_index]
                        mod3_train, mod3_val = original_mod3[train_index], original_mod3[val_index]
                        y_time_train, y_time_val = y_time[train_index], y_time[val_index]
                        y_event_train, y_event_val = y_event[train_index], y_event[val_index]
                        t_train = torch.randint(0, 100, (len(train_index),))
                        t_val = torch.randint(0, 100, (len(val_index),))

                        train_loader = DataLoader(TensorDataset(mod1_train, mod2_train, mod3_train, y_time_train, y_event_train, t_train),
                                                  batch_size=params['batch_size'], shuffle=True)
                        val_loader = DataLoader(TensorDataset(mod1_val, mod2_val, mod3_val, y_time_val, y_event_val, t_val),
                                                batch_size=params['batch_size'], shuffle=False)

                        model = CombinedModel(input_dim1, input_dim2, input_dim3, hidden_dim).to(device)
                        initialize_weights(model)
                        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
                        criterion, contrastive_criterion = DeepSurvLoss(), ContrastiveLoss(temperature=0.5)
                        fold_cindex = []

                        for epoch in range(params["num_epochs"]):
                            model.train()
                            total_loss = 0
                            for mod1_b, mod2_b, mod3_b, y_time_b, y_event_b, t_b in train_loader:
                                mod1_b, mod2_b, mod3_b, y_time_b, y_event_b, t_b = (
                                    mod1_b.to(device), mod2_b.to(device), mod3_b.to(device),
                                    y_time_b.to(device), y_event_b.to(device), t_b.to(device)
                                )
                                optimizer.zero_grad()
                                outputs, feat1, feat2, feat3 = model(mod1_b, mod2_b, mod3_b, t_b)
                                loss_surv = criterion(outputs, y_time_b.float(), y_event_b.float())
                                loss_contrastive = contrastive_criterion(feat1, feat2) + contrastive_criterion(feat1, feat3) + contrastive_criterion(feat2, feat3)
                                loss = loss_surv + alpha_contrastive * loss_contrastive
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                                optimizer.step()
                                total_loss += loss.item()

                            model.eval()
                            preds, t_true, e_true = [], [], []
                            with torch.no_grad():
                                for mod1_b, mod2_b, mod3_b, y_time_b, y_event_b, t_b in val_loader:
                                    out, _, _, _ = model(mod1_b.to(device), mod2_b.to(device), mod3_b.to(device), t_b.to(device))
                                    preds.append(out.cpu().numpy().flatten())
                                    t_true.append(y_time_b.numpy().flatten())
                                    e_true.append(y_event_b.numpy().flatten())

                            preds, t_true, e_true = np.concatenate(preds), np.concatenate(t_true), np.concatenate(e_true)
                            ci = compute_ci(torch.tensor(preds), torch.tensor(t_true), torch.tensor(e_true))
                            fold_cindex.append(ci)
                            save_predictions(fold, epoch, t_true, e_true, preds, os.path.join(output_dir, "predictions"))
                            print(f"Fold {fold + 1}, Epoch {epoch + 1}, C-index: {ci:.4f}")

                        folds_cindex.append(fold_cindex)
                    epoch_means = np.mean(folds_cindex, axis=0)
                    max_mean = np.max(epoch_means)
                    with open(overall_results_file, 'a', newline='') as f:
                        csv.writer(f).writerow([seed, lr, batch_size, hidden_dim, alpha_contrastive, max_mean])
                    print(f"Finished: seed={seed}, best C-index={max_mean:.4f}")

print("ALL FINISHED")
