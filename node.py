from dataclasses import dataclass
from enum import Enum


class NodeType(Enum):
    Unused = 0
    Start = 1
    End = 2
    Connection = 3
    Intersection = 4


@dataclass(eq=False)
class Node:
    id: int
    x: int
    y: int
    type: NodeType = NodeType.Unused

    def __hash__(self):
        return self.id

    def __repr__(self):
        return f"Node(id={self.id}, x={self.x}, y={self.y}, type={self.type.name})"
