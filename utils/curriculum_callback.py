"""
Curriculum learning callback for Stable-Baselines3
Adapts the existing CurriculumManager to SB3's callback interface
"""

from stable_baselines3.common.callbacks import BaseCallback

from utils.curriculum import CurriculumManager


class CurriculumCallback(BaseCallback):
    """
    Callback for curriculum learning with SB3

    Gradually increases environment difficulty during training
    """

    def __init__(self, curriculum_config: dict, verbose: int = 1):
        super().__init__(verbose)
        self.curriculum = CurriculumManager(curriculum_config)
        self.episode_count = 0

    def _on_step(self) -> bool:
        """
        Called at every step

        Returns:
            bool: Whether to continue training
        """
        # Count completed episodes
        if self.locals.get("dones") is not None:
            for done in self.locals["dones"]:
                if done:
                    self.episode_count += 1

        # Update curriculum
        self.curriculum.update(self.episode_count)

        # Update environment if curriculum advanced
        if self.curriculum.should_update_env():
            env_updates = self.curriculum.get_env_updates()

            if self.verbose > 0:
                print(f"\n{'=' * 60}")
                print(f"Curriculum: Advanced to stage '{self.curriculum.current_stage}'")
                print(f"Max aircraft: {env_updates.get('max_aircraft')}")
                print(f"Difficulty: {env_updates.get('difficulty')}")
                print(f"Episodes completed: {self.episode_count}")
                print(f"{'=' * 60}\n")

            # Update environment difficulty
            # Note: This updates max_aircraft in the environment config
            # The actual environment will use this on next reset
            try:
                self.training_env.env_method("set_max_aircraft", env_updates.get("max_aircraft"))
            except AttributeError:
                # Environment doesn't have set_max_aircraft method
                # This is okay - environment will use config value on reset
                pass

            # Log to tensorboard
            if self.logger:
                self.logger.record("curriculum/stage_idx", self.curriculum.current_stage_idx)
                self.logger.record("curriculum/max_aircraft", env_updates.get("max_aircraft"))
                self.logger.record("curriculum/episodes", self.episode_count)

        return True

    def _on_training_end(self) -> None:
        """
        Called at the end of training
        """
        progress = self.curriculum.get_progress()
        if self.verbose > 0:
            print("\nCurriculum Progress:")
            print(
                f"  Final stage: {progress['stage']} ({progress['stage_index'] + 1}/{progress['total_stages']})"
            )
            print(
                f"  Episodes in stage: {progress['episodes_in_stage']}/{progress['target_episodes']}"
            )
