"""
Standard benchmark scenarios for evaluating OpenScope RL agents.

Provides consistent test scenarios for fair comparison across approaches.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np


@dataclass
class BenchmarkScenario:
    """
    A standard test scenario for OpenScope RL evaluation.

    Attributes:
        name: Scenario name
        description: Scenario description
        airport: Airport code (e.g., "KLAS")
        max_aircraft: Maximum number of aircraft
        duration: Episode duration in game seconds
        difficulty: Difficulty level (1-5)
        config: Additional configuration parameters
    """
    name: str
    description: str
    airport: str
    max_aircraft: int
    duration: int
    difficulty: int
    config: Dict[str, Any]


class OpenScopeBenchmark:
    """
    Standard benchmark suite for OpenScope RL agents.

    Provides consistent test scenarios for evaluating and comparing
    different RL approaches.

    Example:
        >>> benchmark = OpenScopeBenchmark()
        >>> scenarios = benchmark.get_scenarios()
        >>> for scenario in scenarios:
        ...     results = evaluate_agent(agent, scenario)
        ...     print(f"{scenario.name}: {results['success_rate']:.2%}")
    """

    def __init__(self):
        self.scenarios = self._create_scenarios()

    def _create_scenarios(self) -> List[BenchmarkScenario]:
        """Create standard benchmark scenarios."""
        return [
            # Easy: Simple arrivals
            BenchmarkScenario(
                name="simple_arrivals",
                description="2-3 aircraft, arrivals only, wide separation",
                airport="KLAS",
                max_aircraft=3,
                duration=300,
                difficulty=1,
                config={
                    "initial_aircraft": 2,
                    "spawn_rate": 0.1,
                    "traffic_type": "arrivals_only",
                }
            ),

            # Medium: Mixed traffic
            BenchmarkScenario(
                name="mixed_traffic",
                description="5-7 aircraft, mixed arrivals/departures",
                airport="KLAS",
                max_aircraft=7,
                duration=600,
                difficulty=2,
                config={
                    "initial_aircraft": 4,
                    "spawn_rate": 0.2,
                    "traffic_type": "mixed",
                }
            ),

            # Hard: Dense traffic
            BenchmarkScenario(
                name="dense_traffic",
                description="10+ aircraft, high conflict probability",
                airport="KLAS",
                max_aircraft=12,
                duration=900,
                difficulty=3,
                config={
                    "initial_aircraft": 8,
                    "spawn_rate": 0.3,
                    "traffic_type": "mixed",
                }
            ),

            # Very Hard: Conflict resolution
            BenchmarkScenario(
                name="conflict_resolution",
                description="Pre-configured conflict scenarios",
                airport="KLAS",
                max_aircraft=8,
                duration=300,
                difficulty=4,
                config={
                    "initial_aircraft": 6,
                    "spawn_rate": 0.0,
                    "traffic_type": "conflict_scenario",
                    "force_conflicts": True,
                }
            ),

            # Expert: High density at major airport
            BenchmarkScenario(
                name="major_airport_peak",
                description="15+ aircraft at busy airport, peak traffic",
                airport="KJFK",  # JFK - busier airport
                max_aircraft=15,
                duration=1200,
                difficulty=5,
                config={
                    "initial_aircraft": 10,
                    "spawn_rate": 0.4,
                    "traffic_type": "mixed",
                }
            ),
        ]

    def get_scenarios(self, difficulty: Optional[int] = None) -> List[BenchmarkScenario]:
        """
        Get benchmark scenarios, optionally filtered by difficulty.

        Args:
            difficulty: Optional difficulty level (1-5) to filter scenarios

        Returns:
            List of benchmark scenarios
        """
        if difficulty is None:
            return self.scenarios
        return [s for s in self.scenarios if s.difficulty == difficulty]

    def get_scenario(self, name: str) -> Optional[BenchmarkScenario]:
        """
        Get a specific scenario by name.

        Args:
            name: Scenario name

        Returns:
            Benchmark scenario or None if not found
        """
        for scenario in self.scenarios:
            if scenario.name == name:
                return scenario
        return None

    def evaluate_agent(self, agent, scenario: BenchmarkScenario,
                      num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate an agent on a benchmark scenario.

        Args:
            agent: RL agent to evaluate
            scenario: Benchmark scenario
            num_episodes: Number of evaluation episodes

        Returns:
            Dictionary of metrics
        """
        from .metrics import MetricsTracker

        tracker = MetricsTracker()

        # Note: This is a template - actual implementation would create
        # an environment configured for the scenario and run episodes

        results = {
            "scenario_name": scenario.name,
            "difficulty": scenario.difficulty,
            "num_episodes": num_episodes,
            "success_rate": 0.0,  # Placeholder
            "avg_violations": 0.0,
            "avg_throughput": 0.0,
            "avg_episode_length": 0.0,
        }

        return results

    def compare_agents(self, agents: Dict[str, Any],
                      scenarios: Optional[List[BenchmarkScenario]] = None,
                      num_episodes: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple agents across benchmark scenarios.

        Args:
            agents: Dictionary of {name: agent} to compare
            scenarios: List of scenarios (default: all scenarios)
            num_episodes: Number of evaluation episodes per scenario

        Returns:
            Dictionary of {agent_name: {scenario_name: metrics}}
        """
        if scenarios is None:
            scenarios = self.scenarios

        results = {}

        for agent_name, agent in agents.items():
            results[agent_name] = {}
            for scenario in scenarios:
                metrics = self.evaluate_agent(agent, scenario, num_episodes)
                results[agent_name][scenario.name] = metrics

        return results

    def print_summary(self, results: Dict[str, Dict[str, float]]) -> None:
        """
        Print a summary of comparison results.

        Args:
            results: Results from compare_agents()
        """
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)

        for agent_name, agent_results in results.items():
            print(f"\n{agent_name}:")
            print("-" * 80)

            for scenario_name, metrics in agent_results.items():
                print(f"  {scenario_name}:")
                print(f"    Success Rate: {metrics.get('success_rate', 0):.2%}")
                print(f"    Avg Violations: {metrics.get('avg_violations', 0):.2f}")
                print(f"    Avg Throughput: {metrics.get('avg_throughput', 0):.2f} aircraft/hour")

        print("\n" + "="*80)
