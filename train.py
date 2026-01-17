import time
import numpy as np
import torch
from environment import PokeGlass
from agent import PokemonAgent

def train_classical_baseline(episodes=10000):
    print(f"Starting Classical Baseline Training for {episodes} episodes...")
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize Environment
    # Headless=True for speed, False to watch (if supported)
    try:
        env = PokeGlass(rom_path='PokemonRed_Classical.gb', headless=True, emulation_speed=0)
    except FileNotFoundError:
        print("Error: 'PokemonRed_Classical.gb' not found. Please run ./battle_royale.sh first to create it.")
        return

    # Initialize Agent
    agent = PokemonAgent(n_features=8, n_actions=8, device=device)
    
    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        loss = 0
        
        start_time = time.time()
        
        while not done and step_count < 1000: # Max steps per episode to prevent getting stuck
            # Select Action
            action = agent.get_action(state)
            
            # Step Environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update Agent
            loss = agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            # Optional: Print progress every 100 steps
            if step_count % 100 == 0:
                print(f"Ep {episode+1} Step {step_count}: Reward={reward:.2f} Loss={loss:.4f} Map={info['map_id']}")
        
        duration = time.time() - start_time
        print(f"Episode {episode+1}/{episodes} | Steps: {step_count} | Total Reward: {total_reward:.2f} | Time: {duration:.1f}s | Epsilon: {agent.epsilon:.3f}")
        
        # Periodic Checkpoint
        if (episode + 1) % 1000 == 0:
            agent.save_checkpoint(f'pokemon_classical_ep{episode+1}.pth')
        
    env.close()
    agent.save_checkpoint('pokemon_classical.pth')
    print("Training Complete.")

if __name__ == "__main__":
    train_classical_baseline(episodes=10000)
