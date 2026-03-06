import torch
import torch.nn as nn
import torch.optim as optim
import time

# --- 1. ТВОЯ УНИКАЛЬНАЯ МОДЕЛЬ (TLIF - 6 параметров) ---
class TLIF_Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Циклическое ядро (3 веса)
        self.w1 = nn.Parameter(torch.ones(1) * 0.5)
        self.w2 = nn.Parameter(torch.ones(1) * 0.5)
        self.w3 = nn.Parameter(torch.ones(1) * 0.5)
        self.w4 = nn.Parameter(torch.ones(1) * 0.5)
        # Проекции (3 веса + 3 смещения = 6)
        self.wx, self.wy, self.wz = [nn.Parameter(torch.ones(1)) for _ in range(3)]
        self.b1, self.b2, self.b3 = [nn.Parameter(torch.zeros(1)) for _ in range(3)]

    def forward(self, x):
        x1, x2 = x[:, 0:1], x[:, 1:2]

        # МУТАЦИЯ: Добавляем возведение в степень (self.W[6] и W[7])
        # Это позволит формуле прогибаться точно под "горб" Планка

        # 1-й отпечаток: Экспоненциальное затухание
        term1 = torch.exp(-x1 * self.wx + self.b1) * (self.w1 * self.w2)

        # 2-й отпечаток: Степенная зависимость (Ключ к Планку!)
        # Используем abs, чтобы не было ошибок с отрицательными числами
        term2 = (torch.abs(x2 + self.b2) ** -self.W[6]) * (self.w1 * self.w3)

        # 3-й отпечаток: Линейная коррекция
        term3 = (x1 * self.wz + self.b3) * (self.w2 * self.w3)

        return term1 + term2 + term3

# --- 2. СТАНДАРТНЫЙ "ЧЕРНЫЙ ЯЩИК" (MLP - ~300 параметров) ---
class MLP_BlackBox(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

# --- 3. ПОДГОТОВКА ДАННЫХ (Спектр Планка) ---
x_raw = torch.linspace(0.1, 3.0, 100)
x1, x2 = torch.meshgrid(x_raw, x_raw, indexing='ij')
x_train = torch.stack([x1.flatten(), x2.flatten()], dim=1)
# Цель: имитация сложной физики
y_target = 1.5 / ((x_train[:, 0:1]**5) * (torch.exp(2.5 / x_train[:, 0:1]) - 1))

# --- 4. ЗАПУСК БЕНЧМАРКА ---
def train_model(model, name):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    start_time = time.time()
    for _ in range(1000):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(x_train), y_target)
        loss.backward()
        optimizer.step()
    duration = time.time() - start_time
    params = sum(p.numel() for p in model.parameters())
    return loss.item(), duration, params

loss_tlif, time_tlif, p_tlif = train_model(TLIF_Net(), "TLIF")
loss_mlp, time_mlp, p_mlp = train_model(MLP_BlackBox(), "MLP")

# --- 5. ВЕРДИКТ ДЛЯ ПУБЛИКАЦИИ ---
print(f"{'Архитектура':<15} | {'Параметры':<10} | {'Ошибка (MSE)':<15} | {'Время (сек)':<10}")
print("-" * 60)
print(f"{'TLIF (Твоя)':<15} | {p_tlif:<10} | {loss_tlif:<15.6f} | {time_tlif:<10.2f}")
print(f"{'MLP (Ящик)':<15} | {p_mlp:<10} | {loss_mlp:<15.6f} | {time_mlp:<10.2f}")

accuracy_gain = (1 - (loss_tlif / (loss_mlp + 1e-8))) * 100
print(f"\nВывод: Твоя формула в {p_mlp // p_tlif} раз компактнее!")
