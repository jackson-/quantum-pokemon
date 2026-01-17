import time
import numpy as np
import torch
from environment import PokeGlass
from agent import PokemonAgent

def train_quantum_agent(episodes=10000):
    print(f"Starting HYBRID QUANTUM-CLASSICAL Training for {episodes} episodes...")
    
    # Quantum simulation is CPU-heavy but PennyLane-Torch interface works best on CPU for small circuits
    # unless using specialized lightning devices. We'll stick to CPU for stability.
    device = "cpu"
    print(f"Using device: {device}")

    # Initialize Environment
    try:
        env = PokeGlass(rom_path='PokemonRed_Quantum.gb', headless=True, emulation_speed=0)
    except FileNotFoundError:
        print("Error: 'PokemonRed_Quantum.gb' not found. Please run ./battle_royale.sh first to create it.")
        return

    # Initialize Agent with Quantum Brain
    # Using 4 features to match 4 qubits for a direct mapping
    agent = PokemonAgent(n_features=4, n_actions=8, brain_type='quantum', learning_rate=1e-2, device=device)
    
    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        
        start_time = time.time()
        
        while not done and step_count < 1000: # Max steps per episode matched to classical
            # Select Action
            action = agent.get_action(state)
            
            # Step Environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update Agent (Hybrid Gradient Descent!)
            # This backpropagates from Classical Brain -> Quantum Circuit -> Classical Eye
            loss = agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if step_count % 10 == 0:
                print(f"Ep {episode+1} Step {step_count}: Reward={reward:.2f} Loss={loss:.4f} Q-Brain Active")
        
        duration = time.time() - start_time
        print(f"Episode {episode+1}/{episodes} | Steps: {step_count} | Total Reward: {total_reward:.2f} | Time: {duration:.1f}s")
        
        # Periodic Checkpoint
        if (episode + 1) % 1000 == 0:
            agent.save_checkpoint(f'pokemon_quantum_ep{episode+1}.pth')

    env.close()
    agent.save_checkpoint('pokemon_quantum.pth')
    print("Quantum Training Complete.")

if __name__ == "__main__":
    train_quantum_agent(episodes=10000)
