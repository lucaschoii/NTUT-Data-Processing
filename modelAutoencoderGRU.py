import torch
import torch.nn as nn
import torch.optim as optim

# Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Define the GRU Model
class GRUPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(GRUPredictor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Use last time step output
        return out

# Hyperparameters
input_dim = 50  # Adjust based on input features
latent_dim = 16
hidden_dim = 32
output_dim = 1  # Binary classification for tremor onset
num_layers = 1

# Instantiate models
autoencoder = Autoencoder(input_dim, latent_dim)
gru = GRUPredictor(latent_dim, hidden_dim, output_dim, num_layers)

# Example data
batch_size = 32
sequence_length = 10  # Number of time steps
x = torch.randn(batch_size, sequence_length, input_dim)  # Simulated noisy input

# Train autoencoder first
encoded, _ = autoencoder(x.view(-1, input_dim))  # Flatten for autoencoder
encoded = encoded.view(batch_size, sequence_length, -1)  # Reshape for GRU

# Predict tremor onset using GRU
predictions = gru(encoded)
print(predictions.shape)  # Should output (batch_size, output_dim)
