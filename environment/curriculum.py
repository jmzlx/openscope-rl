"""
Curriculum learning management for OpenScope RL training.

This module implements a progressive curriculum that gradually increases
task difficulty by adding more aircraft and adjusting environment parameters.
This approach dramatically improves sample efficiency and final performance
compared to training on full difficulty from the start.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class CurriculumStage:
    """Configuration for a single curriculum stage."""
    
    name: str
    max_aircraft: int
    min_score_threshold: float
    timesteps: int
    episode_length: int = 600
    success_threshold: float = 0.5  # Advance when 50%+ success rate
    min_avg_reward: float = -2000.0  # Minimum average reward to advance
    
    def __post_init__(self):
        """Validate stage configuration."""
        if self.max_aircraft < 1:
            raise ValueError("max_aircraft must be at least 1")
        if self.timesteps < 1:
            raise ValueError("timesteps must be at least 1")
        if not 0.0 <= self.success_threshold <= 1.0:
            raise ValueError("success_threshold must be between 0.0 and 1.0")


class CurriculumManager:
    """
    Manages curriculum learning progression through difficulty stages.
    
    The curriculum starts with simple scenarios (few aircraft) and gradually
    increases complexity as the agent demonstrates competence.
    
    Example:
        >>> curriculum = CurriculumManager()
        >>> for stage in curriculum.stages:
        ...     train_on_stage(stage)
        ...     if curriculum.should_advance(eval_results):
        ...         print(f"✅ Advancing from {stage.name}")
    """
    
    def __init__(self, custom_stages: Optional[List[CurriculumStage]] = None):
        """
        Initialize curriculum manager.
        
        Args:
            custom_stages: Optional custom stage definitions. If None, uses default stages.
        """
        if custom_stages is not None:
            self.stages = custom_stages
        else:
            self.stages = self._create_default_stages()
        
        self.current_stage_idx = 0
        self.stage_history: List[Dict[str, Any]] = []
        
        logger.info(f"Curriculum initialized with {len(self.stages)} stages")
    
    def _create_default_stages(self) -> List[CurriculumStage]:
        """Create default curriculum stages."""
        return [
            CurriculumStage(
                name="easy",
                max_aircraft=3,
                min_score_threshold=-5000,
                timesteps=100000,
                episode_length=300,
                success_threshold=0.5,
                min_avg_reward=-1500.0,
            ),
            CurriculumStage(
                name="medium",
                max_aircraft=5,
                min_score_threshold=-8000,
                timesteps=100000,
                episode_length=400,
                success_threshold=0.45,
                min_avg_reward=-2000.0,
            ),
            CurriculumStage(
                name="hard",
                max_aircraft=8,
                min_score_threshold=-12000,
                timesteps=150000,
                episode_length=500,
                success_threshold=0.4,
                min_avg_reward=-2500.0,
            ),
            CurriculumStage(
                name="expert",
                max_aircraft=12,
                min_score_threshold=-20000,
                timesteps=150000,
                episode_length=600,
                success_threshold=0.35,
                min_avg_reward=-3000.0,
            ),
        ]
    
    @property
    def current_stage(self) -> CurriculumStage:
        """Get the current curriculum stage."""
        return self.stages[self.current_stage_idx]
    
    def get_stage(self, name: str) -> Optional[CurriculumStage]:
        """
        Get stage by name.
        
        Args:
            name: Stage name to retrieve
            
        Returns:
            CurriculumStage if found, None otherwise
        """
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None
    
    def should_advance(self, evaluation_results: Dict[str, float]) -> bool:
        """
        Determine if agent should advance to next stage.
        
        Args:
            evaluation_results: Dictionary with metrics:
                - 'success_rate': Fraction of successful episodes (0.0-1.0)
                - 'avg_reward': Average episode reward
                - 'avg_violations': Average separation violations per episode
                
        Returns:
            True if agent meets advancement criteria
        """
        if self.current_stage_idx >= len(self.stages) - 1:
            logger.info("Already at final stage")
            return False
        
        stage = self.current_stage
        success_rate = evaluation_results.get('success_rate', 0.0)
        avg_reward = evaluation_results.get('avg_reward', -float('inf'))
        avg_violations = evaluation_results.get('avg_violations', float('inf'))
        
        # Criteria for advancement:
        # 1. Meet success rate threshold
        # 2. Meet minimum average reward
        # 3. Keep violations low (<5 per episode on average)
        meets_success = success_rate >= stage.success_threshold
        meets_reward = avg_reward >= stage.min_avg_reward
        meets_safety = avg_violations < 5.0
        
        can_advance = meets_success and meets_reward and meets_safety
        
        logger.info(
            f"Stage '{stage.name}' advancement check: "
            f"success_rate={success_rate:.2%} (threshold={stage.success_threshold:.2%}), "
            f"avg_reward={avg_reward:.1f} (threshold={stage.min_avg_reward:.1f}), "
            f"avg_violations={avg_violations:.1f} (threshold=5.0) "
            f"→ {'ADVANCE' if can_advance else 'CONTINUE'}"
        )
        
        return can_advance
    
    def advance(self) -> bool:
        """
        Move to the next curriculum stage.
        
        Returns:
            True if advanced, False if already at final stage
        """
        if self.current_stage_idx >= len(self.stages) - 1:
            logger.warning("Cannot advance: already at final stage")
            return False
        
        old_stage = self.current_stage.name
        self.current_stage_idx += 1
        new_stage = self.current_stage.name
        
        logger.info(f"✅ Advanced from '{old_stage}' to '{new_stage}'")
        return True
    
    def record_stage_completion(
        self,
        evaluation_results: Dict[str, float],
        training_metrics: Dict[str, Any],
    ) -> None:
        """
        Record metrics for completed stage.
        
        Args:
            evaluation_results: Final evaluation metrics
            training_metrics: Training statistics for the stage
        """
        stage_record = {
            'stage_name': self.current_stage.name,
            'stage_index': self.current_stage_idx,
            'evaluation': evaluation_results,
            'training': training_metrics,
        }
        
        self.stage_history.append(stage_record)
        logger.info(f"Recorded completion of stage '{self.current_stage.name}'")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of curriculum training progress.
        
        Returns:
            Dictionary with curriculum statistics
        """
        return {
            'total_stages': len(self.stages),
            'completed_stages': len(self.stage_history),
            'current_stage': self.current_stage.name,
            'current_stage_index': self.current_stage_idx,
            'stage_history': self.stage_history,
        }
    
    def reset(self) -> None:
        """Reset curriculum to first stage."""
        self.current_stage_idx = 0
        self.stage_history = []
        logger.info("Curriculum reset to first stage")
    
    def print_curriculum(self) -> None:
        """Print curriculum stages in human-readable format."""
        print("\n" + "="*80)
        print("CURRICULUM STAGES")
        print("="*80)
        
        for i, stage in enumerate(self.stages):
            marker = "→" if i == self.current_stage_idx else " "
            status = "CURRENT" if i == self.current_stage_idx else ""
            if i < self.current_stage_idx:
                status = "COMPLETED"
            
            print(f"\n{marker} Stage {i+1}: {stage.name.upper()} {status}")
            print(f"   Aircraft: {stage.max_aircraft}")
            print(f"   Episode Length: {stage.episode_length} steps")
            print(f"   Training Steps: {stage.timesteps:,}")
            print(f"   Score Threshold: {stage.min_score_threshold}")
            print(f"   Success Rate Target: {stage.success_threshold:.0%}")
            print(f"   Avg Reward Target: {stage.min_avg_reward:.0f}")
        
        print("\n" + "="*80)
        print(f"Progress: {len(self.stage_history)}/{len(self.stages)} stages completed")
        print("="*80 + "\n")


def create_simple_curriculum(
    max_aircraft_list: List[int],
    timesteps_per_stage: int = 100000,
) -> CurriculumManager:
    """
    Create a simple curriculum with custom aircraft counts.
    
    Args:
        max_aircraft_list: List of aircraft counts for each stage
        timesteps_per_stage: Training steps for each stage
        
    Returns:
        CurriculumManager configured with custom stages
        
    Example:
        >>> curriculum = create_simple_curriculum([2, 4, 6, 10], timesteps_per_stage=50000)
    """
    stages = []
    
    for i, max_aircraft in enumerate(max_aircraft_list):
        # Scale thresholds with aircraft count
        min_score_threshold = -2000 * max_aircraft
        episode_length = min(300 + i * 100, 600)
        
        stage = CurriculumStage(
            name=f"stage_{i+1}",
            max_aircraft=max_aircraft,
            min_score_threshold=min_score_threshold,
            timesteps=timesteps_per_stage,
            episode_length=episode_length,
            success_threshold=max(0.3, 0.6 - i * 0.05),  # Gradually lower expectations
            min_avg_reward=-1000 * max_aircraft,
        )
        stages.append(stage)
    
    return CurriculumManager(custom_stages=stages)

