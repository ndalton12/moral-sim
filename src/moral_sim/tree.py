import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import math


@dataclass
class WeightedOutcome:
    next_node: str
    probability: float


@dataclass
class Choice:
    text: str
    next_node: Optional[str] = None  # Becomes optional
    probabilistic_outcomes: Optional[List[WeightedOutcome]] = None  # Added

    def __post_init__(self):
        if self.next_node is None and self.probabilistic_outcomes is None:
            raise ValueError(
                f"Choice '{self.text}' must have either 'next_node' or 'probabilistic_outcomes'."
            )
        if self.next_node is not None and self.probabilistic_outcomes is not None:
            raise ValueError(
                f"Choice '{self.text}' cannot have both 'next_node' and 'probabilistic_outcomes'."
            )
        if self.probabilistic_outcomes:
            total_prob = sum(
                outcome.probability for outcome in self.probabilistic_outcomes
            )
            if not math.isclose(total_prob, 1.0, abs_tol=1e-9):
                raise ValueError(
                    f"Probabilities for choice '{self.text}' must sum to 1.0 (currently sum to {total_prob})."
                )


@dataclass
class Node:
    id: str
    node_type: str  # 'decision' or 'outcome'
    prompt: Optional[str] = None  # Only for decision nodes
    choices: Optional[List[Choice]] = None  # Only for decision nodes
    description: Optional[str] = None  # Only for outcome nodes
    outcomes: Optional[Dict[str, float]] = None  # Only for outcome nodes


@dataclass
class DecisionTree:
    start_node: str
    nodes: Dict[str, Node] = field(default_factory=dict)

    @classmethod
    def load_from_yaml(cls, file_path: str) -> "DecisionTree":
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        nodes_data = data.get("nodes", {})
        nodes_dict = {}
        for node_id, node_info in nodes_data.items():
            choices = None
            if node_info.get("choices"):
                choices_list = []
                for choice_data in node_info["choices"]:
                    prob_outcomes_data = choice_data.get("probabilistic_outcomes")
                    prob_outcomes = None
                    if prob_outcomes_data:
                        prob_outcomes = [
                            WeightedOutcome(**po_data) for po_data in prob_outcomes_data
                        ]

                    choices_list.append(
                        Choice(
                            text=choice_data["text"],
                            next_node=choice_data.get("next_node"),
                            probabilistic_outcomes=prob_outcomes,
                        )
                    )
                choices = choices_list

            nodes_dict[node_id] = Node(
                id=node_id,
                node_type=node_info["type"],
                prompt=node_info.get("prompt"),
                choices=choices,
                description=node_info.get("description"),
                outcomes=node_info.get("outcomes"),
            )

        return cls(start_node=data["start_node"], nodes=nodes_dict)

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)
