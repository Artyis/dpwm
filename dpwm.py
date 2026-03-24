"""
Dual-Plane World Model — Прототип
==================================
Идея: два вектора (физика + язык) складываются в общий эмбеддинг.
Лосс проверяет что операция обратима:
  [физика] + [язык] = [общий]
  [общий]  - [язык] = [физика]  ✓
  [общий]  - [физика] = [язык]  ✓

Физика симулируется как координаты падающего мяча.
Язык — простые эмбеддинги описаний ("мяч падает", "мяч летит" и т.д.)

Запуск:
  pip install torch numpy
  python dual_plane_prototype.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ─── Конфиг ───────────────────────────────────────────────────────────────────
EMBED_DIM = 32  # размер каждого вектора
HIDDEN_DIM = 64  # размер скрытого слоя
EPOCHS = 500
LR = 1e-3
BATCH_SIZE = 16
VOCAB_SIZE = 8  # количество разных текстовых описаний


# ─── Симуляция физики (падающий мяч) ─────────────────────────────────────────
def simulate_ball(n_samples: int) -> torch.Tensor:
    """
    Генерирует траектории мяча.
    Возвращает тензор [n_samples, 4]:
      (x, y, velocity_x, velocity_y)
    """
    t = torch.rand(n_samples)  # момент времени 0..1
    x = 0.5 + 0.3 * torch.cos(2 * torch.pi * t)  # круговая траектория
    y = 1.0 - 0.5 * t**2  # падение вниз
    vx = -0.3 * 2 * torch.pi * torch.sin(2 * torch.pi * t)
    vy = -t
    return torch.stack([x, y, vx, vy], dim=1)  # [n, 4]


def physics_label(physics: torch.Tensor) -> torch.Tensor:
    """
    Простая метка для физики на следующий шаг:
    просто сдвигаем y вниз (мяч продолжает падать).
    """
    next_physics = physics.clone()
    next_physics[:, 1] = next_physics[:, 1] - 0.1  # y уменьшается
    return next_physics


# ─── Словарь текстов ──────────────────────────────────────────────────────────
DESCRIPTIONS = [
    "мяч падает вниз",
    "мяч летит вправо",
    "мяч замедляется",
    "мяч ускоряется",
    "мяч у земли",
    "мяч высоко",
    "мяч движется",
    "мяч остановился",
]


def text_to_index(n_samples: int) -> torch.Tensor:
    """Случайно выбираем описание для каждого примера."""
    return torch.randint(0, VOCAB_SIZE, (n_samples,))


# ─── Архитектура ──────────────────────────────────────────────────────────────


class PhysicsEncoder(nn.Module):
    """Кодирует физические координаты в вектор смысла."""

    def __init__(self, input_dim=4, embed_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, embed_dim),
        )

    def forward(self, x):
        return self.net(x)  # [batch, embed_dim]


class LanguageEncoder(nn.Module):
    """Кодирует индекс текста в языковой вектор."""

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, idx):
        return self.embedding(idx)  # [batch, embed_dim]


class PhysicsDecoder(nn.Module):
    """Восстанавливает физику из общего эмбеддинга."""

    def __init__(self, embed_dim=EMBED_DIM, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class LanguageDecoder(nn.Module):
    """Восстанавливает языковой вектор из общего эмбеддинга."""

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class DualPlaneWorldModel(nn.Module):
    """
    Главная модель.

    Прямой проход:
      physics_vec = PhysicsEncoder(physics)
      lang_vec    = LanguageEncoder(text_idx)
      combined    = physics_vec + lang_vec          ← сложение

    Обратный проход (проверка обратимости):
      recovered_physics_vec = combined - lang_vec   → DecodePhysics
      recovered_lang_vec    = combined - physics_vec → DecodeLang
    """

    def __init__(self):
        super().__init__()
        self.physics_encoder = PhysicsEncoder()
        self.language_encoder = LanguageEncoder()
        self.physics_decoder = PhysicsDecoder()
        self.language_decoder = LanguageDecoder()

    def forward(self, physics, text_idx):
        # Кодируем обе плоскости
        physics_vec = self.physics_encoder(physics)  # [B, D]
        lang_vec = self.language_encoder(text_idx)  # [B, D]

        # Складываем → общий эмбеддинг
        combined = physics_vec + lang_vec  # [B, D]

        # Восстанавливаем каждую плоскость вычитанием
        recovered_physics = self.physics_decoder(combined - lang_vec)
        recovered_lang = self.language_decoder(combined - physics_vec)

        return {
            "physics_vec": physics_vec,
            "lang_vec": lang_vec,
            "combined": combined,
            "recovered_physics": recovered_physics,
            "recovered_lang": recovered_lang,
        }


# ─── Обучение ─────────────────────────────────────────────────────────────────


def train():
    model = DualPlaneWorldModel()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()

    print("=" * 55)
    print("  Dual-Plane World Model — обучение")
    print("=" * 55)
    print(f"  Эпох: {EPOCHS} | Батч: {BATCH_SIZE} | Dim: {EMBED_DIM}")
    print("=" * 55)

    for epoch in range(EPOCHS):
        # ── Генерируем батч ──
        physics = simulate_ball(BATCH_SIZE)  # [B, 4]
        text_idx = text_to_index(BATCH_SIZE)  # [B]
        next_phys = physics_label(physics)  # [B, 4] — цель для физики

        out = model(physics, text_idx)

        # ── Три лосса ──────────────────────────────────────────────────────────

        # Лосс 1: предсказание следующего физического состояния из combined
        predicted_next = model.physics_decoder(out["combined"])
        loss_physics = mse(predicted_next, next_phys)

        # Лосс 2: восстановление физики (combined - lang = physics)
        loss_recover_physics = mse(out["recovered_physics"], next_phys)

        # Лосс 3: восстановление языка (combined - physics = lang)
        loss_recover_lang = mse(
            out["recovered_lang"],
            out["lang_vec"].detach(),  # целевой языковой вектор
        )

        # Общий лосс
        loss = loss_physics + loss_recover_physics + loss_recover_lang

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(
                f"  Эпоха {epoch + 1:4d} | "
                f"Общий: {loss.item():.4f} | "
                f"Физика: {loss_physics.item():.4f} | "
                f"Восст.физика: {loss_recover_physics.item():.4f} | "
                f"Восст.язык: {loss_recover_lang.item():.4f}"
            )

    print("=" * 55)
    print("  Обучение завершено!")
    print("=" * 55)
    return model


# ─── Тест обратимости ─────────────────────────────────────────────────────────


def test_reversibility(model):
    """
    Главный тест идеи:
    показываем незнакомую физику без языка →
    модель должна восстановить физику только из combined.
    """
    print("\n  ТЕСТ: обратимость эмбеддингов")
    print("-" * 55)

    model.eval()
    with torch.no_grad():
        physics = simulate_ball(4)
        text_idx = text_to_index(4)
        out = model(physics, text_idx)

        # Восстанавливаем физику вычитанием языка
        recovered = model.physics_decoder(out["combined"] - out["lang_vec"])
        next_phys = physics_label(physics)
        error = torch.abs(recovered - next_phys).mean().item()

        print(f"  Средняя ошибка восстановления физики: {error:.4f}")

        if error < 0.1:
            print("  ✓ Обратимость работает — идея доказана!")
        elif error < 0.3:
            print("  ~ Частично работает — нужно больше эпох")
        else:
            print("  ✗ Не сработало — нужна доработка архитектуры")

    print("-" * 55)


# ─── Точка входа ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    model = train()
    test_reversibility(model)
