"""
Curriculum learning manager for progressive difficulty
"""



class CurriculumManager:
    """
    Manages curriculum learning progression
    Gradually increases difficulty as agent improves
    """

    def __init__(self, curriculum_config: dict):
        self.config = curriculum_config
        self.stages = curriculum_config["stages"]
        self.current_stage_idx = 0
        self.current_stage = self.stages[0]["name"]
        self.episodes_in_stage = 0
        self.should_update = False

    def update(self, total_episodes: int):
        """Update curriculum based on training progress"""
        current_stage_config = self.stages[self.current_stage_idx]
        target_episodes = current_stage_config["episodes"]

        self.episodes_in_stage = total_episodes - sum(
            stage["episodes"] for stage in self.stages[: self.current_stage_idx]
        )

        # Check if we should advance to next stage
        if self.episodes_in_stage >= target_episodes:
            if self.current_stage_idx < len(self.stages) - 1:
                self.current_stage_idx += 1
                self.current_stage = self.stages[self.current_stage_idx]["name"]
                self.episodes_in_stage = 0
                self.should_update = True
                print(f"\n{'='*60}")
                print(f"Curriculum: Advanced to stage '{self.current_stage}'")
                print(f"Max aircraft: {self.stages[self.current_stage_idx]['max_aircraft']}")
                print(f"Difficulty: {self.stages[self.current_stage_idx]['difficulty']}")
                print(f"{'='*60}\n")

    def should_update_env(self) -> bool:
        """Check if environment should be updated"""
        if self.should_update:
            self.should_update = False
            return True
        return False

    def get_env_updates(self) -> dict:
        """Get environment configuration updates for current stage"""
        current_config = self.stages[self.current_stage_idx]
        return {
            "max_aircraft": current_config["max_aircraft"],
            "difficulty": current_config["difficulty"],
        }

    def get_progress(self) -> dict:
        """Get current curriculum progress"""
        return {
            "stage": self.current_stage,
            "stage_index": self.current_stage_idx,
            "total_stages": len(self.stages),
            "episodes_in_stage": self.episodes_in_stage,
            "target_episodes": self.stages[self.current_stage_idx]["episodes"],
        }
