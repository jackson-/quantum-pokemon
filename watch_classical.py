import sys
import time
import torch
from environment import PokeGlass
from agent import PokemonAgent

def watch_classical(rom_path='PokemonRed_Classical.gb'):
    print(f"Initializing Classical Agent in Watch Mode with ROM: {rom_path}")
    print("A PyBoy window should appear shortly.")
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize Environment with headless=False to see the window
    # emulation_speed=1 is real-time. Set to 0 for max speed, or higher for fast-forward.
    try:
        env = PokeGlass(rom_path=rom_path, headless=False, emulation_speed=1)
    except FileNotFoundError:
        print(f"Error: '{rom_path}' not found.")
        return

    # Initialize Agent
    agent = PokemonAgent(n_features=8, n_actions=8, device=device)
    agent.load_checkpoint('pokemon_classical.pth')
    
    # Run for 1 episode
    state, info = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    
    try:
        while not done:
            # Render is handled by the window, but we need to keep the loop running
            
            # Select Action
            # Use training=False to rely on learned weights
            action = agent.get_action(state, training=False) 
            
            # Step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if step_count % 50 == 0:
                print(f"Step {step_count}: Map={info['map_id']} Reward={reward}")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        env.close()
        print(f"Session Ended. Total Steps: {step_count} | Total Reward: {total_reward}")

if __name__ == "__main__":
    rom = sys.argv[1] if len(sys.argv) > 1 else 'PokemonRed_Classical.gb'
    watch_classical(rom)
