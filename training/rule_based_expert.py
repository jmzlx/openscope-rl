"""
Rule-based expert controller for generating demonstration data.

This module implements a simple heuristic controller that generates reasonable
trajectories for behavioral cloning pre-training. The expert focuses on:
1. Maintaining safe separation between aircraft
2. Guiding aircraft toward exit points
3. Issuing ILS commands when aircraft are near runways
4. Managing altitude and speed efficiently

The expert doesn't need to be perfect - it just needs to be better than random
to provide useful demonstrations for behavioral cloning.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pickle
from pathlib import Path
from dataclasses import dataclass, field

from poc.atc_rl import Realistic3DATCEnv
from poc.atc_rl.constants import SEPARATION_LATERAL_NM, AIRSPACE_SIZE


logger = logging.getLogger(__name__)


@dataclass
class Demonstration:
    """A single demonstration trajectory."""
    observations: List[Dict[str, np.ndarray]] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    total_reward: float = 0.0
    success: bool = False
    length: int = 0


class RuleBasedExpert:
    """
    Rule-based expert controller for ATC.

    This expert uses simple heuristics to control aircraft:
    - Avoid conflicts by changing headings
    - Guide aircraft to exits based on their position
    - Issue landing commands when appropriate
    - Manage altitude for separation

    Example:
        >>> env = Realistic3DATCEnv(max_aircraft=5)
        >>> expert = RuleBasedExpert(env)
        >>> obs, info = env.reset()
        >>> action = expert.get_action(obs)
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """

    def __init__(
        self,
        env: Realistic3DATCEnv,
        conflict_threshold: float = SEPARATION_LATERAL_NM + 2.0,
        exit_threshold: float = 3.0,
        landing_distance_threshold: float = 10.0,
    ):
        """
        Initialize rule-based expert.

        Args:
            env: ATC environment
            conflict_threshold: Distance threshold for conflict avoidance (nm)
            exit_threshold: Distance threshold for exit guidance (nm)
            landing_distance_threshold: Distance threshold for landing (nm)
        """
        self.env = env
        self.conflict_threshold = conflict_threshold
        self.exit_threshold = exit_threshold
        self.landing_distance_threshold = landing_distance_threshold

        # Action space dimensions
        self.max_aircraft = env.max_aircraft
        self.action_shape = (5,)  # [aircraft_id, command_type, altitude, heading, speed]

    def get_action(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate expert action based on observations.

        Args:
            obs: Environment observation

        Returns:
            Action array [aircraft_id, command_type, altitude, heading, speed]
        """
        aircraft_data = obs['aircraft']
        aircraft_mask = obs['aircraft_mask']
        conflict_matrix = obs['conflict_matrix']

        # Find active aircraft
        active_indices = np.where(aircraft_mask)[0]
        if len(active_indices) == 0:
            # No aircraft - return no-op action
            return np.array([self.max_aircraft, 0, 0, 0, 0], dtype=np.int64)

        # Prioritize aircraft based on:
        # 1. Conflicts (highest priority)
        # 2. Proximity to exit/landing
        # 3. Aircraft that haven't been commanded recently

        priorities = self._calculate_priorities(aircraft_data, aircraft_mask, conflict_matrix)
        selected_idx = active_indices[np.argmax(priorities[active_indices])]

        # Generate command for selected aircraft
        action = self._generate_command(selected_idx, aircraft_data, conflict_matrix)

        return action

    def _calculate_priorities(
        self,
        aircraft_data: np.ndarray,
        aircraft_mask: np.ndarray,
        conflict_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate priority scores for each aircraft.

        Args:
            aircraft_data: Aircraft feature matrix
            aircraft_mask: Active aircraft mask
            conflict_matrix: Conflict matrix

        Returns:
            Priority scores for each aircraft
        """
        n_aircraft = len(aircraft_mask)
        priorities = np.zeros(n_aircraft)

        for idx in range(n_aircraft):
            if not aircraft_mask[idx]:
                continue

            # Check for conflicts
            has_conflict = np.any(conflict_matrix[idx, :] > 0)
            if has_conflict:
                priorities[idx] += 100.0

            # Check proximity to edges (potential exit)
            x, y = aircraft_data[idx, 0], aircraft_data[idx, 1]
            dist_to_edge = min(x, y, AIRSPACE_SIZE - x, AIRSPACE_SIZE - y)
            if dist_to_edge < self.exit_threshold:
                priorities[idx] += 50.0

            # Add some randomness to avoid always selecting the same aircraft
            priorities[idx] += np.random.uniform(0, 10)

        return priorities

    def _generate_command(
        self,
        aircraft_idx: int,
        aircraft_data: np.ndarray,
        conflict_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Generate command for a specific aircraft.

        Args:
            aircraft_idx: Index of aircraft to command
            aircraft_data: Aircraft feature matrix
            conflict_matrix: Conflict matrix

        Returns:
            Action array [aircraft_id, command_type, altitude, heading, speed]
        """
        # Extract aircraft state
        # Features: [x, y, altitude, heading, speed, vx, vy, target_alt, target_hdg,
        #            target_spd, is_landing, runway, dist_to_runway, time_active]
        x, y = aircraft_data[aircraft_idx, 0], aircraft_data[aircraft_idx, 1]
        altitude = aircraft_data[aircraft_idx, 2]
        heading = aircraft_data[aircraft_idx, 3]
        speed = aircraft_data[aircraft_idx, 4]
        is_landing = aircraft_data[aircraft_idx, 10]
        dist_to_runway = aircraft_data[aircraft_idx, 12]

        # Check for conflicts
        has_conflict = np.any(conflict_matrix[aircraft_idx, :] > 0)

        # Decision logic
        if has_conflict:
            # Conflict avoidance: change heading
            return self._avoid_conflict(aircraft_idx, aircraft_data, conflict_matrix)

        # Check if near runway and should land
        if dist_to_runway < self.landing_distance_threshold and not is_landing:
            # Issue ILS command
            return np.array([aircraft_idx, 3, 0, 0, 0], dtype=np.int64)  # command_type=3 is ILS

        # Check if near edge - prepare for exit
        dist_to_edge = min(x, y, AIRSPACE_SIZE - x, AIRSPACE_SIZE - y)
        if dist_to_edge < self.exit_threshold:
            # Guide toward nearest exit
            return self._guide_to_exit(aircraft_idx, aircraft_data)

        # Default: adjust altitude for separation (create vertical layers)
        return self._adjust_altitude(aircraft_idx, aircraft_data)

    def _avoid_conflict(
        self,
        aircraft_idx: int,
        aircraft_data: np.ndarray,
        conflict_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Generate heading change to avoid conflicts.

        Args:
            aircraft_idx: Index of aircraft
            aircraft_data: Aircraft feature matrix
            conflict_matrix: Conflict matrix

        Returns:
            Heading change action
        """
        # Find conflicting aircraft
        conflicting_indices = np.where(conflict_matrix[aircraft_idx, :] > 0)[0]

        if len(conflicting_indices) == 0:
            return np.array([aircraft_idx, 1, 0, 6, 0], dtype=np.int64)  # No change

        # Calculate average direction to conflicts
        my_pos = aircraft_data[aircraft_idx, :2]
        conflict_positions = aircraft_data[conflicting_indices, :2]

        # Vector pointing away from conflicts
        to_conflicts = conflict_positions - my_pos
        avg_to_conflict = np.mean(to_conflicts, axis=0)

        # Turn away from conflicts
        if np.linalg.norm(avg_to_conflict) > 0:
            # Calculate perpendicular direction
            perpendicular = np.array([-avg_to_conflict[1], avg_to_conflict[0]])

            # Choose left or right turn based on which is safer
            if np.random.random() > 0.5:
                perpendicular = -perpendicular

            # Map to heading change index (index 6 is no change)
            # Larger changes for closer conflicts
            heading_change_idx = 8  # +20 degrees (index 8 in heading_changes)
        else:
            heading_change_idx = 6  # No change

        # Return heading command
        return np.array([aircraft_idx, 1, 0, heading_change_idx, 0], dtype=np.int64)

    def _guide_to_exit(
        self,
        aircraft_idx: int,
        aircraft_data: np.ndarray,
    ) -> np.ndarray:
        """
        Generate command to guide aircraft to exit.

        Args:
            aircraft_idx: Index of aircraft
            aircraft_data: Aircraft feature matrix

        Returns:
            Heading change action toward exit
        """
        x, y = aircraft_data[aircraft_idx, 0], aircraft_data[aircraft_idx, 1]
        heading = aircraft_data[aircraft_idx, 3]

        # Determine which edge is closest
        distances = {
            'north': AIRSPACE_SIZE - y,
            'south': y,
            'east': AIRSPACE_SIZE - x,
            'west': x,
        }

        closest_edge = min(distances, key=distances.get)

        # Target heading for each edge
        target_headings = {
            'north': 0,    # 0 degrees
            'east': 90,    # 90 degrees
            'south': 180,  # 180 degrees
            'west': 270,   # 270 degrees
        }

        target_heading = target_headings[closest_edge]

        # Calculate heading change needed
        heading_diff = (target_heading - heading + 180) % 360 - 180

        # Map to discrete heading change index
        heading_changes = [-90, -60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60, 90]
        heading_change_idx = np.argmin(np.abs(np.array(heading_changes) - heading_diff))

        # Return heading command
        return np.array([aircraft_idx, 1, 0, heading_change_idx, 0], dtype=np.int64)

    def _adjust_altitude(
        self,
        aircraft_idx: int,
        aircraft_data: np.ndarray,
    ) -> np.ndarray:
        """
        Adjust altitude to create vertical separation layers.

        Args:
            aircraft_idx: Index of aircraft
            aircraft_data: Aircraft feature matrix

        Returns:
            Altitude change action
        """
        current_altitude = aircraft_data[aircraft_idx, 2]

        # Create altitude layers at 5000 ft intervals
        altitude_layers = [10, 50, 90, 130, 170]  # Indices for 10k, 50k, 90k, 130k, 170k ft

        # Find closest layer
        current_alt_idx = int(current_altitude / 1000 / 10)  # Convert to index
        current_alt_idx = np.clip(current_alt_idx, 0, 17)

        # If not at a standard layer, move to one
        target_layer_idx = min(altitude_layers, key=lambda x: abs(x - current_alt_idx))

        # Return altitude command
        return np.array([aircraft_idx, 0, target_layer_idx, 0, 0], dtype=np.int64)


def generate_demonstrations(
    n_episodes: int = 1000,
    max_aircraft: int = 5,
    episode_length: int = 1000,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> List[Demonstration]:
    """
    Generate demonstration episodes using the rule-based expert.

    Args:
        n_episodes: Number of episodes to generate
        max_aircraft: Maximum number of aircraft in environment
        episode_length: Maximum episode length
        save_path: Path to save demonstrations (optional)
        verbose: Whether to print progress

    Returns:
        List of Demonstration objects
    """
    env = Realistic3DATCEnv(
        max_aircraft=max_aircraft,
        episode_length=episode_length,
        render_mode=None,
    )
    expert = RuleBasedExpert(env)

    demonstrations = []

    for episode_idx in range(n_episodes):
        obs, info = env.reset()
        demo = Demonstration()

        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            # Get expert action
            action = expert.get_action(obs)

            # Store transition
            demo.observations.append({
                'aircraft': obs['aircraft'].copy(),
                'aircraft_mask': obs['aircraft_mask'].copy(),
                'global_state': obs['global_state'].copy(),
                'conflict_matrix': obs['conflict_matrix'].copy(),
            })
            demo.actions.append(action)

            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)

            demo.rewards.append(reward)
            total_reward += reward

        # Store episode statistics
        demo.total_reward = total_reward
        demo.success = info.get('successful_landings', 0) > 0
        demo.length = len(demo.actions)

        demonstrations.append(demo)

        if verbose and (episode_idx + 1) % 100 == 0:
            avg_reward = np.mean([d.total_reward for d in demonstrations[-100:]])
            success_rate = np.mean([d.success for d in demonstrations[-100:]])
            logger.info(f"Episode {episode_idx + 1}/{n_episodes} - "
                       f"Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.2%}")

    # Save demonstrations if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump(demonstrations, f)

        logger.info(f"Saved {len(demonstrations)} demonstrations to {save_path}")

    return demonstrations


def load_demonstrations(load_path: str) -> List[Demonstration]:
    """
    Load demonstrations from file.

    Args:
        load_path: Path to demonstrations file

    Returns:
        List of Demonstration objects
    """
    with open(load_path, 'rb') as f:
        demonstrations = pickle.load(f)

    logger.info(f"Loaded {len(demonstrations)} demonstrations from {load_path}")
    return demonstrations


def print_demonstration_stats(demonstrations: List[Demonstration]) -> None:
    """
    Print statistics about demonstrations.

    Args:
        demonstrations: List of Demonstration objects
    """
    total_transitions = sum(d.length for d in demonstrations)
    avg_reward = np.mean([d.total_reward for d in demonstrations])
    std_reward = np.std([d.total_reward for d in demonstrations])
    success_rate = np.mean([d.success for d in demonstrations])
    avg_length = np.mean([d.length for d in demonstrations])

    print("\n" + "="*60)
    print("Demonstration Statistics")
    print("="*60)
    print(f"Number of episodes: {len(demonstrations)}")
    print(f"Total transitions: {total_transitions}")
    print(f"Average episode length: {avg_length:.1f}")
    print(f"Average reward: {avg_reward:.2f} +/- {std_reward:.2f}")
    print(f"Success rate: {success_rate:.2%}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Generate demonstrations
    print("Generating expert demonstrations...")
    demonstrations = generate_demonstrations(
        n_episodes=1000,
        max_aircraft=5,
        episode_length=1000,
        save_path="data/expert_demonstrations.pkl",
        verbose=True,
    )

    # Print statistics
    print_demonstration_stats(demonstrations)
