import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Denoising Diffusion ----------
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

# ---------- Transformer ----------
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

# ---------- DeepSurv ----------
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

# ---------- ContrastiveLoss ----------
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

# ---------- CombinedModel ----------
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
