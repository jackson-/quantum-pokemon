import sys
import time
import torch
from environment import PokeGlass
from agent import PokemonAgent

def watch_quantum(rom_path='PokemonRed_Quantum.gb'):
    print(f"Initializing QUANTUM Agent in Watch Mode with ROM: {rom_path}")
    print("A PyBoy window should appear shortly.")
    print("NOTE: Inference will be slower than classical due to Quantum Simulation.")
    
    # Check device (Quantum usually runs on CPU for simulation)
    device = "cpu"
    print(f"Using device: {device}")

    # Initialize Environment with headless=False to see the window
    # emulation_speed=1 is real-time.
    try:
        env = PokeGlass(rom_path=rom_path, headless=False, emulation_speed=1)
    except FileNotFoundError:
        print(f"Error: '{rom_path}' not found.")
        return

    # Initialize Agent with Quantum Brain
    # Must match training config: n_features=4
    agent = PokemonAgent(n_features=4, n_actions=8, brain_type='quantum', device=device)
    agent.load_checkpoint('pokemon_quantum.pth')
    
    # Run for 1 episode
    state, info = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    
    try:
        while not done:
            # Select Action
            # Use training=False to rely on learned weights
            action = agent.get_action(state, training=False) 
            
            # Step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if step_count % 10 == 0:
                print(f"Step {step_count}: Map={info['map_id']} Reward={reward} | Quantum Inference Complete")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        env.close()
        print(f"Session Ended. Total Steps: {step_count} | Total Reward: {total_reward}")

if __name__ == "__main__":
    rom = sys.argv[1] if len(sys.argv) > 1 else 'PokemonRed_Quantum.gb'
    watch_quantum(rom)
