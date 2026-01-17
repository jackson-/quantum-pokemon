import unittest
from unittest.mock import MagicMock
import numpy as np
import torch
from agent import PokemonAgent
# We won't import PokeGlass here to avoid PyBoy dependency for this mock test

class MockEnv:
    def __init__(self):
        self.observation_space = type('obj', (object,), {'shape': (144, 160, 3)})
        self.action_space = type('obj', (object,), {'n': 8})
    
    def reset(self):
        return np.zeros((144, 160, 3), dtype=np.uint8), {}
    
    def step(self, action):
        next_state = np.zeros((144, 160, 3), dtype=np.uint8)
        reward = 1.0
        terminated = False
        truncated = False
        info = {"map_id": 0}
        return next_state, reward, terminated, truncated, info

class TestClassicalBaseline(unittest.TestCase):
    def test_agent_forward_pass(self):
        print("\nTesting Agent Forward Pass...")
        device = "cpu"
        agent = PokemonAgent(n_features=8, n_actions=8, device=device)
        
        # Create dummy observation (144, 160, 3)
        obs = np.zeros((144, 160, 3), dtype=np.uint8)
        
        action = agent.get_action(obs, training=False)
        self.assertTrue(0 <= action < 8)
        print(f"Agent selected action: {action}")

    def test_training_loop_mock(self):
        print("\nTesting Training Loop Logic (Mock)...")
        env = MockEnv()
        agent = PokemonAgent(n_features=8, n_actions=8, device="cpu")
        
        state, _ = env.reset()
        action = agent.get_action(state)
        next_state, reward, done, _, _ = env.step(action)
        
        loss = agent.update(state, action, reward, next_state, done)
        self.assertIsInstance(loss, float)
        print(f"Update step successful. Loss: {loss}")

if __name__ == '__main__':
    unittest.main()
