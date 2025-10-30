"""
Training Data Collector for OpenScope RL.

This module extracts all data needed for training ATC models, including:
- Observations (normalized for RL)
- Actions
- Rewards
- Episode trajectories
- Metadata (conflicts, violations, etc.)

Compatible with:
- PPO/RL training
- Behavioral Cloning
- Decision Transformer
- Trajectory Transformer
"""

import json
import logging
import pickle
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm

from environment import PlaywrightEnv, create_default_config
from environment.state_processor import StateProcessor
from environment.reward_calculator import RewardCalculator, create_reward_calculator

logger = logging.getLogger(__name__)


@dataclass
class TrainingTransition:
    """Single transition for training data."""
    
    # Observation (normalized)
    observation: Dict[str, np.ndarray]
    
    # Action taken
    action: Dict[str, int]
    
    # Reward received
    reward: float
    
    # Next observation
    next_observation: Optional[Dict[str, np.ndarray]] = None
    
    # Done flag
    done: bool = False
    
    # Info dict
    info: Dict[str, Any] = None
    
    # Raw state (for analysis/debugging)
    raw_state: Dict[str, Any] = None
    
    # Timestamp
    timestamp: float = 0.0


@dataclass
class TrainingEpisode:
    """Complete episode for training."""
    
    episode_id: str
    
    # Episode trajectory
    observations: List[Dict[str, np.ndarray]]
    actions: List[Dict[str, int]]
    rewards: List[float]
    dones: List[bool]
    infos: List[Dict[str, Any]]
    
    # Raw states (optional, for full data capture)
    raw_states: Optional[List[Dict[str, Any]]] = None
    
    # Metadata
    episode_length: int = 0
    total_reward: float = 0.0
    final_score: int = 0
    violations: int = 0
    conflicts_encountered: int = 0
    successful_landings: int = 0
    airport: str = ""
    max_aircraft: int = 0
    
    # Computed fields (for offline RL)
    returns_to_go: Optional[np.ndarray] = None
    
    def compute_returns_to_go(self):
        """Compute return-to-go for Decision Transformer."""
        rtgs = []
        rtg = 0.0
        
        for reward in reversed(self.rewards):
            rtg += reward
            rtgs.insert(0, rtg)
        
        self.returns_to_go = np.array(rtgs, dtype=np.float32)


class TrainingDataCollector:
    """
    Collect training data from OpenScope environment.
    
    This collector extracts all data needed for training:
    - Normalized observations (matching StateProcessor output)
    - Actions in environment format
    - Rewards
    - Episode trajectories
    - Raw state data (optional)
    
    Example:
        >>> collector = TrainingDataCollector(
        ...     output_dir="training_data",
        ...     max_aircraft=10
        ... )
        >>> episodes = collector.collect_episodes(
        ...     num_episodes=100,
        ...     policy="random"
        ... )
    """
    
    def __init__(
        self,
        output_dir: str = "training_data",
        max_aircraft: int = 20,
        airport: str = "KLAS",
        timewarp: int = 5,
        headless: bool = False,
        save_raw_states: bool = False,  # Save raw state for analysis
        save_wind_data: bool = True,    # Extract wind components
        reward_strategy: str = "default",
    ):
        """
        Initialize training data collector.
        
        Args:
            output_dir: Directory to save collected data
            max_aircraft: Maximum number of aircraft
            airport: ICAO airport code
            timewarp: Time acceleration factor
            headless: Whether to run browser in headless mode
            save_raw_states: Whether to save raw game states (increases size)
            save_wind_data: Whether to extract wind components via methods
            reward_strategy: Reward calculation strategy
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_aircraft = max_aircraft
        self.airport = airport
        self.timewarp = timewarp
        self.headless = headless
        self.save_raw_states = save_raw_states
        self.save_wind_data = save_wind_data
        
        # Create config
        config = create_default_config(
            airport=airport,
            max_aircraft=max_aircraft,
            timewarp=timewarp,
            headless=headless,
        )
        
        self.config = config
        self.state_processor = StateProcessor(config)
        self.reward_calculator = create_reward_calculator(
            config.reward_config, reward_strategy
        )
        
        logger.info(f"TrainingDataCollector initialized: {airport}, {max_aircraft} aircraft")
        logger.info(f"Output directory: {self.output_dir}")
    
    def collect_episode(
        self,
        episode_id: str,
        policy: str = "random",
        max_steps: int = 1000,
    ) -> TrainingEpisode:
        """
        Collect a single episode of training data.
        
        Args:
            episode_id: Unique identifier for this episode
            policy: Policy to use ("random", "heuristic", "expert")
            max_steps: Maximum steps per episode
            
        Returns:
            TrainingEpisode with all data
        """
        logger.info(f"Collecting episode: {episode_id}")
        
        # Create environment
        env = PlaywrightEnv(
            airport=self.airport,
            max_aircraft=self.max_aircraft,
            timewarp=self.timewarp,
            headless=self.headless,
        )
        
        try:
            # Reset environment
            obs, info = env.reset()
            
            # Initialize episode data
            observations = []
            actions = []
            rewards = []
            dones = []
            infos = []
            raw_states = [] if self.save_raw_states else None
            
            prev_state = info.get("raw_state", {})
            prev_raw_state = prev_state.copy()
            
            # Episode statistics
            violations = 0
            conflicts_encountered = set()
            
            # Collect trajectory
            for step in range(max_steps):
                # Get current raw state for reward calculation
                current_state = info.get("raw_state", {})
                
                # Save observation
                observations.append({
                    key: val.copy() if isinstance(val, np.ndarray) else val
                    for key, val in obs.items()
                })
                
                # Get action from policy
                action = self._select_action(env, obs, policy, current_state)
                actions.append(action.copy() if isinstance(action, dict) else action)
                
                # Step environment (reward is already calculated by environment)
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                done = terminated or truncated
                
                rewards.append(float(reward))
                dones.append(done)
                infos.append(next_info.copy())
                
                # Save raw state if requested
                if self.save_raw_states:
                    # Extract enhanced state with all available data
                    raw_state = self._extract_complete_state(
                        env.browser_manager.page,
                        current_state
                    )
                    raw_states.append(raw_state)
                
                # Track violations and conflicts
                for conflict in current_state.get("conflicts", []):
                    if conflict.get("hasViolation"):
                        violations += 1
                    conflict_key = (
                        conflict.get("aircraft1", ""),
                        conflict.get("aircraft2", "")
                    )
                    if conflict_key[0] and conflict_key[1]:
                        conflicts_encountered.add(conflict_key)
                
                # Update for next iteration
                prev_raw_state = current_state.copy()
                obs = next_obs
                info = next_info
                
                if done:
                    break
            
            # Get final statistics
            final_score = info.get("raw_state", {}).get("score", 0)
            final_state = info.get("raw_state", {})
            
            # Create episode
            episode = TrainingEpisode(
                episode_id=episode_id,
                observations=observations,
                actions=actions,
                rewards=rewards,
                dones=dones,
                infos=infos,
                raw_states=raw_states,
                episode_length=len(rewards),
                total_reward=sum(rewards),
                final_score=final_score,
                violations=violations,
                conflicts_encountered=len(conflicts_encountered),
                airport=self.airport,
                max_aircraft=self.max_aircraft,
            )
            
            # Compute returns-to-go for offline RL
            episode.compute_returns_to_go()
            
            logger.info(f"Episode {episode_id} collected: {episode.episode_length} steps, "
                       f"reward={episode.total_reward:.2f}, violations={episode.violations}")
            
            return episode
            
        finally:
            env.close()
    
    def collect_episodes(
        self,
        num_episodes: int,
        policy: str = "random",
        max_steps: int = 1000,
        save_individual: bool = True,
    ) -> List[TrainingEpisode]:
        """
        Collect multiple episodes.
        
        Args:
            num_episodes: Number of episodes to collect
            policy: Policy to use
            max_steps: Maximum steps per episode
            save_individual: Whether to save each episode individually
            
        Returns:
            List of TrainingEpisode objects
        """
        episodes = []
        
        for i in tqdm(range(num_episodes), desc="Collecting episodes"):
            episode_id = f"ep_{i:05d}_{policy}"
            
            try:
                episode = self.collect_episode(
                    episode_id=episode_id,
                    policy=policy,
                    max_steps=max_steps,
                )
                episodes.append(episode)
                
                # Save individual episode if requested
                if save_individual:
                    self._save_episode(episode)
                    
            except Exception as e:
                logger.error(f"Failed to collect episode {episode_id}: {e}")
                continue
        
        # Save dataset summary
        self._save_dataset_summary(episodes)
        
        logger.info(f"Collected {len(episodes)} episodes")
        return episodes
    
    def _extract_complete_state(
        self,
        page,
        base_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract complete state including wind and other data.
        
        Args:
            page: Playwright page object
            base_state: Base state from environment
            
        Returns:
            Enhanced state dictionary
        """
        enhanced_state = base_state.copy()
        
        if not self.save_wind_data:
            return enhanced_state
        
        # Extract wind components via JavaScript
        js_extract_wind = """
        (function() {
            if (!window.aircraftController || !window.aircraftController.aircraft) {
                return {};
            }
            
            const aircraftList = window.aircraftController.aircraft.list;
            const windData = {};
            
            for (let i = 0; i < Math.min(aircraftList.length, 20); i++) {
                const ac = aircraftList[i];
                const callsign = ac.callsign;
                
                if (typeof ac.getWindComponents === 'function') {
                    try {
                        windData[callsign] = ac.getWindComponents();
                    } catch (e) {
                        windData[callsign] = null;
                    }
                }
            }
            
            return windData;
        })();
        """
        
        try:
            wind_data = page.evaluate(js_extract_wind)
            enhanced_state["wind_components"] = wind_data
        except Exception as e:
            logger.debug(f"Failed to extract wind data: {e}")
            enhanced_state["wind_components"] = {}
        
        return enhanced_state
    
    def _select_action(
        self,
        env: PlaywrightEnv,
        obs: Dict[str, np.ndarray],
        policy: str,
        state: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Select action based on policy.
        
        Args:
            env: Environment
            obs: Current observation
            policy: Policy name
            state: Raw state
            
        Returns:
            Action dictionary
        """
        if policy == "random":
            return env.action_space.sample()
        
        elif policy == "heuristic":
            return self._heuristic_action(obs, state)
        
        elif policy == "expert":
            # TODO: Use rule-based expert if available
            return env.action_space.sample()
        
        else:
            logger.warning(f"Unknown policy: {policy}, using random")
            return env.action_space.sample()
    
    def _heuristic_action(
        self,
        obs: Dict[str, np.ndarray],
        state: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Simple heuristic policy.
        
        Args:
            obs: Observation
            state: Raw state
            
        Returns:
            Action dictionary
        """
        # Simple heuristic: if conflicts exist, try to resolve them
        conflicts = state.get("conflicts", [])
        aircraft_list = state.get("aircraft", [])
        
        if conflicts and aircraft_list:
            # Target first conflict
            conflict = conflicts[0]
            ac1_callsign = conflict.get("aircraft1")
            
            # Find aircraft index
            for i, ac in enumerate(aircraft_list):
                if ac.get("callsign") == ac1_callsign:
                    # Issue altitude command to separate
                    current_alt = ac.get("altitude", 0)
                    new_alt = min(current_alt + 1000, 40000)
                    alt_idx = min(int(new_alt / 1000), 40)
                    
                    # Command type index: 0=altitude, 1=heading, 2=speed, 3=ils, 4=direct
                    return {
                        "aircraft_id": min(i, self.max_aircraft),
                        "command_type": 0,  # ALTITUDE
                        "altitude": alt_idx,
                        "heading": 0,
                        "speed": 0,
                    }
        
        # Default: no-op
        return {
            "aircraft_id": self.max_aircraft,  # No-op
            "command_type": 0,
            "altitude": 0,
            "heading": 0,
            "speed": 0,
        }
    
    def _save_episode(self, episode: TrainingEpisode):
        """Save individual episode to file."""
        episode_file = self.output_dir / f"{episode.episode_id}.pkl"
        
        # Convert numpy arrays to lists for JSON serialization if needed
        episode_dict = asdict(episode)
        
        # Convert numpy arrays
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        episode_dict = convert_numpy(episode_dict)
        
        with open(episode_file, 'wb') as f:
            pickle.dump(episode_dict, f)
    
    def _save_dataset_summary(self, episodes: List[TrainingEpisode]):
        """Save dataset summary statistics."""
        if not episodes:
            return
        
        summary = {
            "num_episodes": len(episodes),
            "total_transitions": sum(ep.episode_length for ep in episodes),
            "avg_episode_length": np.mean([ep.episode_length for ep in episodes]),
            "avg_reward": np.mean([ep.total_reward for ep in episodes]),
            "total_violations": sum(ep.violations for ep in episodes),
            "total_conflicts": sum(ep.conflicts_encountered for ep in episodes),
            "airport": self.airport,
            "max_aircraft": self.max_aircraft,
            "observation_info": self.state_processor.get_observation_info(),
        }
        
        summary_file = self.output_dir / "dataset_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved dataset summary to {summary_file}")


def load_training_episodes(data_dir: str) -> List[TrainingEpisode]:
    """
    Load training episodes from directory.
    
    Args:
        data_dir: Directory containing episode files
        
    Returns:
        List of TrainingEpisode objects
    """
    data_path = Path(data_dir)
    episode_files = sorted(data_path.glob("ep_*.pkl"))
    
    episodes = []
    for ep_file in episode_files:
        with open(ep_file, 'rb') as f:
            episode_dict = pickle.load(f)
            # Convert back to TrainingEpisode
            episode = TrainingEpisode(**episode_dict)
            episodes.append(episode)
    
    logger.info(f"Loaded {len(episodes)} episodes from {data_dir}")
    return episodes

