import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class ClassicalEye(nn.Module):
    """
    The Eye: A CNN feature extractor.
    Input: (B, 3, 144, 160) image
    Output: A vector of 'n_features' (e.g., 8)
    """
    def __init__(self, n_features=8):
        super(ClassicalEye, self).__init__()
        # Input shape: (3, 144, 160)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4, padding=0), # -> (16, 35, 39)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0), # -> (32, 16, 18)
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate flat size: 32 * 16 * 18 = 9216
        self.fc = nn.Sequential(
            nn.Linear(9216, 256),
            nn.ReLU(),
            nn.Linear(256, n_features),
            nn.Tanh() # Normalize features to [-1, 1] for better Quantum compatibility later
        )

    def forward(self, x):
        x = self.cnn(x)
        features = self.fc(x)
        return features

from circuit import QuantumCircuit

class ClassicalBrain(nn.Module):
    """
    The Classical Baseline Brain.
    Input: 'n_features' from The Eye.
    Output: Action values/probabilities for 8 actions.
    """
    def __init__(self, n_features=8, n_actions=8):
        super(ClassicalBrain, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )

    def forward(self, features):
        return self.network(features)

class QuantumBrain(nn.Module):
    """
    The Quantum Brain (The Brain).
    Input: 'n_features' from The Eye.
    Output: Action values for 8 actions via a Variational Quantum Circuit.
    """
    def __init__(self, n_features=4, n_actions=8, n_layers=2):
        super(QuantumBrain, self).__init__()
        # Initialize the Quantum Circuit
        # We'll use 4 qubits as defined in the technical specs
        n_qubits = 4
        self.qc = QuantumCircuit(n_qubits=n_qubits, n_layers=n_layers)
        self.q_layer = self.qc.get_torch_layer()
        
        # If n_features from Eye doesn't match n_qubits, we need a bottleneck
        self.bottleneck = nn.Linear(n_features, n_qubits) if n_features != n_qubits else nn.Identity()
        
        # Final "Hybrid" layer to map 4 Qubit outputs to 8 Actions
        self.post_process = nn.Linear(n_qubits, n_actions)

    def forward(self, features):
        # 1. Classical Pre-processing (Bottleneck)
        x = self.bottleneck(features)
        
        # 2. Quantum Processing (VQC)
        # Inputs to AngleEmbedding are usually scaled to [0, pi] or [-pi, pi]
        # Our Eye uses Tanh, so outputs are [-1, 1]. Scaling by pi:
        q_out = self.q_layer(x * np.pi)
        
        # 3. Post-processing to Actions
        return self.post_process(q_out)

class PokemonAgent:
    """
    The full agent combining Eye and Brain.
    """
    def __init__(self, n_features=8, n_actions=8, brain_type='classical', learning_rate=1e-3, device='cpu'):
        self.device = device
        self.n_actions = n_actions
        self.brain_type = brain_type
        
        # Components
        self.eye = ClassicalEye(n_features).to(device)
        
        if brain_type == 'quantum':
            self.brain = QuantumBrain(n_features, n_actions).to(device)
        else:
            self.brain = ClassicalBrain(n_features, n_actions).to(device)
        
        # Optimizers
        self.optimizer = optim.Adam(
            list(self.eye.parameters()) + list(self.brain.parameters()), 
            lr=learning_rate
        )
        
        # Loss function (MSE for DQN-like, CrossEntropy for classification/policy)
        # We'll stick to a simple DQN approach for stability in Phase 2
        self.criterion = nn.MSELoss()
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def get_action(self, observation, training=True):
        """
        observation: numpy array (144, 160, 3)
        """
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        # Preprocess
        # Transpose to (3, 144, 160) and normalize
        state = torch.FloatTensor(observation).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        
        with torch.no_grad():
            features = self.eye(state)
            q_values = self.brain(features)
            
        return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state, done):
        """
        Simple Single-Step Q-Learning Update (for baseline testing).
        For production, use Replay Buffer + Batch Training.
        """
        # Prepare Tensors
        state_t = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        next_state_t = torch.FloatTensor(next_state).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        action_t = torch.LongTensor([action]).to(self.device)
        reward_t = torch.FloatTensor([reward]).to(self.device)
        done_t = torch.FloatTensor([done]).to(self.device)
        
        # Forward Pass
        features = self.eye(state_t)
        q_values = self.brain(features)
        current_q = q_values.gather(1, action_t.unsqueeze(1)).squeeze(1)
        
        # Target
        with torch.no_grad():
            next_features = self.eye(next_state_t)
            next_q = self.brain(next_features)
            max_next_q = next_q.max(1)[0]
            target_q = reward_t + (0.99 * max_next_q * (1 - done_t))
            
        # Loss & Step
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()

    def save_checkpoint(self, filename):
        checkpoint = {
            'eye': self.eye.state_dict(),
            'brain': self.brain.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, filename)
        print(f"Model saved to {filename}")

    def load_checkpoint(self, filename):
        try:
            checkpoint = torch.load(filename, map_location=self.device)
            self.eye.load_state_dict(checkpoint['eye'])
            self.brain.load_state_dict(checkpoint['brain'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            print(f"Model loaded from {filename}")
        except FileNotFoundError:
            print(f"No checkpoint found at {filename}. Starting from scratch.")
