import numpy as np
import matplotlib.pyplot as plt

# Example parameters
initial_lr = 0.1
epochs = 100
decay_rate = 0.02
step_size = 20
gamma = 0.5

# 1. Constant Learning Rate
def constant_lr(epoch, initial_lr):
    return initial_lr

# 2. Time-based Decay
def time_based_decay(epoch, initial_lr, decay_rate):
    return initial_lr / (1 + decay_rate * epoch)

# 3. Step Decay
def step_decay(epoch, initial_lr, step_size, gamma):
    return initial_lr * (gamma ** (epoch // step_size))

# 4. Exponential Decay
def exponential_decay(epoch, initial_lr, decay_rate):
    return initial_lr * np.exp(-decay_rate * epoch)

# 5. Polynomial Decay
def polynomial_decay(epoch, initial_lr, max_epochs, power=1.0):
    return initial_lr * (1 - epoch / max_epochs) ** power

# 6. Cosine Annealing
def cosine_annealing(epoch, initial_lr, max_epochs):
    return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / max_epochs))

# 7. Warm Restart (Cosine with restarts)
def cosine_warm_restart(epoch, initial_lr, restart_period):
    return initial_lr * 0.5 * (1 + np.cos(np.pi * (epoch % restart_period) / restart_period))

# Generate learning rates for all schedules
epochs_array = np.arange(0, epochs)

lr_constant = [constant_lr(epoch, initial_lr) for epoch in epochs_array]
lr_time_decay = [time_based_decay(epoch, initial_lr, decay_rate) for epoch in epochs_array]
lr_step_decay = [step_decay(epoch, initial_lr, step_size, gamma) for epoch in epochs_array]
lr_exp_decay = [exponential_decay(epoch, initial_lr, decay_rate) for epoch in epochs_array]
lr_poly_decay = [polynomial_decay(epoch, initial_lr, epochs, power=2.0) for epoch in epochs_array]
lr_cosine = [cosine_annealing(epoch, initial_lr, epochs) for epoch in epochs_array]
lr_warm_restart = [cosine_warm_restart(epoch, initial_lr, 25) for epoch in epochs_array]

# Plot all schedules
plt.figure(figsize=(15, 10))

plt.subplot(2, 4, 1)
plt.plot(epochs_array, lr_constant, 'b-', linewidth=2)
plt.title('Constant Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.grid(True)

plt.subplot(2, 4, 2)
plt.plot(epochs_array, lr_time_decay, 'r-', linewidth=2)
plt.title('Time-based Decay')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.grid(True)

plt.subplot(2, 4, 3)
plt.plot(epochs_array, lr_step_decay, 'g-', linewidth=2)
plt.title('Step Decay')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.grid(True)

plt.subplot(2, 4, 4)
plt.plot(epochs_array, lr_exp_decay, 'm-', linewidth=2)
plt.title('Exponential Decay')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.grid(True)

plt.subplot(2, 4, 5)
plt.plot(epochs_array, lr_poly_decay, 'c-', linewidth=2)
plt.title('Polynomial Decay')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.grid(True)

plt.subplot(2, 4, 6)
plt.plot(epochs_array, lr_cosine, 'orange', linewidth=2)
plt.title('Cosine Annealing')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.grid(True)

plt.subplot(2, 4, 7)
plt.plot(epochs_array, lr_warm_restart, 'purple', linewidth=2)
plt.title('Cosine Warm Restart')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.grid(True)

plt.subplot(2, 4, 8)
plt.plot(epochs_array, lr_constant, 'b-', label='Constant', alpha=0.7)
plt.plot(epochs_array, lr_time_decay, 'r-', label='Time Decay', alpha=0.7)
plt.plot(epochs_array, lr_step_decay, 'g-', label='Step Decay', alpha=0.7)
plt.plot(epochs_array, lr_exp_decay, 'm-', label='Exponential', alpha=0.7)
plt.plot(epochs_array, lr_cosine, 'orange', label='Cosine', alpha=0.7)
plt.title('Comparison of All Schedules')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print some example values
print("Learning Rate Schedule Examples:")
print("="*50)
print(f"Epoch\tConstant\tTime Decay\tStep Decay\tExponential\tCosine")
print("-"*70)
for epoch in [0, 10, 20, 30, 50, 75, 99]:
    const = constant_lr(epoch, initial_lr)
    time_d = time_based_decay(epoch, initial_lr, decay_rate)
    step_d = step_decay(epoch, initial_lr, step_size, gamma)
    exp_d = exponential_decay(epoch, initial_lr, decay_rate)
    cos_d = cosine_annealing(epoch, initial_lr, epochs)
    print(f"{epoch}\t{const:.4f}\t\t{time_d:.4f}\t\t{step_d:.4f}\t\t{exp_d:.4f}\t\t{cos_d:.4f}")

# Example implementation in a training loop
print("\n" + "="*50)
print("Example Training Loop with Learning Rate Schedule:")
print("="*50)

class LearningRateScheduler:
    def __init__(self, schedule_type='step', initial_lr=0.01, **kwargs):
        self.schedule_type = schedule_type
        self.initial_lr = initial_lr
        self.params = kwargs
        self.epoch = 0
    
    def get_lr(self):
        if self.schedule_type == 'constant':
            return self.initial_lr
        elif self.schedule_type == 'step':
            step_size = self.params.get('step_size', 10)
            gamma = self.params.get('gamma', 0.5)
            return self.initial_lr * (gamma ** (self.epoch // step_size))
        elif self.schedule_type == 'exponential':
            decay_rate = self.params.get('decay_rate', 0.02)
            return self.initial_lr * np.exp(-decay_rate * self.epoch)
        elif self.schedule_type == 'cosine':
            max_epochs = self.params.get('max_epochs', 100)
            return self.initial_lr * 0.5 * (1 + np.cos(np.pi * self.epoch / max_epochs))
    
    def step(self):
        self.epoch += 1

# Example usage
scheduler = LearningRateScheduler('step', initial_lr=0.1, step_size=10, gamma=0.5)

print("Training simulation with Step Decay:")
for epoch in range(25):
    current_lr = scheduler.get_lr()
    print(f"Epoch {epoch:2d}: Learning Rate = {current_lr:.6f}")
    
    # Simulate training step here
    # optimizer.param_groups[0]['lr'] = current_lr
    # ... training code ...

    scheduler.step()

print("\nPyTorch/TensorFlow Implementation Examples:")
print("="*50)
print("""
# PyTorch Examples:
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR

optimizer = optim.SGD(model.parameters(), lr=0.1)

# Step decay
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# Exponential decay
scheduler = ExponentialLR(optimizer, gamma=0.95)

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=100)

# In training loop:
for epoch in range(num_epochs):
    # Training code here
    scheduler.step()

# TensorFlow/Keras Examples:
import tensorflow as tf

# Exponential decay
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1,
    decay_steps=100,
    decay_rate=0.96,
    staircase=True
)

# Cosine decay
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.1,
    decay_steps=1000
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
""")