import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import io

class PokeGlass(gym.Env):
    def __init__(self, rom_path='PokemonRed.gb', headless=True, emulation_speed=0):
        super(PokeGlass, self).__init__()
        
        # Initialize PyBoy
        window_type = "null" if headless else "SDL2"
        self.pyboy = PyBoy(rom_path, window=window_type)
        self.pyboy.set_emulation_speed(emulation_speed)
        
        # Save initial state for reset
        self.init_state = io.BytesIO()
        self.pyboy.save_state(self.init_state)
        self.init_state.seek(0)
        
        # Define Action Space: 8 buttons
        # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: A, 5: B, 6: START, 7: SELECT
        self.action_space = spaces.Discrete(8)
        
        # Mapping actions to PyBoy WindowEvents
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
            WindowEvent.PRESS_BUTTON_SELECT
        ]
        
        self.release_actions = [
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START,
            WindowEvent.RELEASE_BUTTON_SELECT
        ]

        # Define Observation Space: 160x144 grayscale screen (1 channel) or RGB (3 channels)
        # PyBoy screen is 160x144. We'll return the full RGB array for the CNN to process.
        self.observation_space = spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8)
        
        # Memory Addresses (from project specs)
        self.MEM_X_COORD = 0xD362
        self.MEM_Y_COORD = 0xD361
        self.MEM_MAP_ID  = 0xD35E
        self.MEM_PARTY_COUNT = 0xD163
        self.MEM_P1_HP_CURR = 0xD16C
        self.MEM_P1_HP_MAX = 0xD16E
        self.MEM_ENEMY_HP = 0xCFE6
        self.MEM_BADGES = 0xD356
        self.MEM_P1_LEVEL = 0xD18C # Common address for P1 Level

        # State tracking for rewards
        self.seen_maps = set()
        self.seen_coords = set()
        self.total_reward = 0
        self.last_enemy_hp = -1
        self.last_p1_level = -1
        
    def step(self, action):
        # 1. Perform Action
        self.pyboy.send_input(self.valid_actions[action])
        # Tick for 24 frames (approx 400ms) to allow action to complete
        for _ in range(24): 
            self.pyboy.tick()
        self.pyboy.send_input(self.release_actions[action])

        # 2. Capture Observation
        observation = self.pyboy.screen.ndarray[:, :, :3]
        
        # 3. Calculate Reward & Read Memory
        info = self._get_info()
        reward = self._calculate_reward(info)
        self.total_reward += reward
        
        # 4. Check Termination
        # For now, we don't have a hard termination condition other than crashing or manual stop
        terminated = False 
        truncated = False
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset emulator to initial state
        self.init_state.seek(0)
        self.pyboy.load_state(self.init_state)
        
        # Reset state tracking
        self.seen_maps = set()
        self.seen_coords = set()
        self.total_reward = 0
        self.last_enemy_hp = -1
        self.last_p1_level = -1
        
        # Initial observation
        observation = self.pyboy.screen.ndarray[:, :, :3]
        info = self._get_info()
        
        # Initialize 'last' values to avoid instant rewards on reset
        self.seen_maps.add(info['map_id'])
        self.seen_coords.add((info['x_pos'], info['y_pos'], info['map_id']))
        self.last_enemy_hp = info['enemy_hp']
        self.last_p1_level = info['p1_level']
        
        return observation, info

    def render(self):
        pass

    def close(self):
        self.pyboy.stop()

    def _get_info(self):
        """
        Reads directly from the emulator's memory.
        """
        return {
            "x_pos": self.pyboy.memory[self.MEM_X_COORD],
            "y_pos": self.pyboy.memory[self.MEM_Y_COORD],
            "map_id": self.pyboy.memory[self.MEM_MAP_ID],
            "party_count": self.pyboy.memory[self.MEM_PARTY_COUNT],
            "p1_hp": self.pyboy.memory[self.MEM_P1_HP_CURR],
            "p1_max_hp": self.pyboy.memory[self.MEM_P1_HP_MAX],
            "enemy_hp": self.pyboy.memory[self.MEM_ENEMY_HP],
            "badges": self.pyboy.memory[self.MEM_BADGES],
            "p1_level": self.pyboy.memory[self.MEM_P1_LEVEL]
        }

    def _calculate_reward(self, info):
        reward = 0.0
        
        # 1. Exploration Reward (New Map IDs)
        if info['map_id'] not in self.seen_maps:
            self.seen_maps.add(info['map_id'])
            reward += 1.0 # Significant reward for exploration

        # 2. Tile Exploration Reward (New X,Y coords)
        # Gives a small nudge to keep moving
        coord = (info['x_pos'], info['y_pos'], info['map_id'])
        if coord not in self.seen_coords:
            self.seen_coords.add(coord)
            reward += 0.05 
            
        # 3. Battle Reward (Winning)
        # If enemy HP goes from >0 to 0, we likely won a battle
        # Note: This is a simplification. Enemy HP might be 0 when no battle is active.
        # A more robust check might involve checking battle state flags, but this is a start.
        if self.last_enemy_hp > 0 and info['enemy_hp'] == 0:
            reward += 5.0 # Big reward for winning
        self.last_enemy_hp = info['enemy_hp']
            
        # 3. Level Up Reward
        if self.last_p1_level > 0 and info['p1_level'] > self.last_p1_level:
            reward += 10.0 # Huge reward for leveling up
        self.last_p1_level = info['p1_level']
        
        return reward

if __name__ == "__main__":
    # Simple test to verify memory reading
    print("Initializing PokeGlass environment...")
    try:
        # Note: You need a valid 'PokemonRed.gb' in the directory for this to run
        env = PokeGlass(headless=True)
        obs, info = env.reset()
        print(f"Initial State - Position: ({info['x_pos']}, {info['y_pos']}), Map ID: {info['map_id']}")
        env.close()
        print("Environment created and closed successfully.")
    except Exception as e:
        print(f"Failed to initialize (likely missing ROM): {e}")
