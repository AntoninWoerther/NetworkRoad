import matplotlib.pyplot as plt
from node import Node, NodeType
from segment import Segment

GRID_SIZE = (20, 10)
def setup_window():
    plt.rcParams["toolbar"] = "None"
    background = "#191E26"
    columns, rows = GRID_SIZE
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    fig.canvas.manager.set_window_title("Road Network")
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)
    ax.set_xlim(-1, columns)
    ax.set_ylim(-1, rows)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0.05, top=1)
    return fig, ax
class Node:
    def __init__(self, id, x, y, node_type=NodeType.Unused):
    self.id = id
    self.x = x
    self.y = y
    self.type = node_type

_node(ax, node):
    colors = {
    NodeType.Unused: "#303844",
    NodeType.Start: "#3CBE82",
    NodeType.End: "#E65A50",
    NodeType.Connection: "#DCD7C3",
    NodeType.Intersection: "#F5B83D",
    }
    ax.plot(node.x, node.y, marker="o", color=colors[node.type], markersize=8, zorder=2)

def draw_segment(ax, segment):
 ax.plot([segment.start.x, segment.end.x], [segment.start.y, segment.end.y], color="#DCD7C3", linewidth=2, zorder=1)


def show_window():
    plt.show()
if __name__ == "__main__":
    fig, ax = setup_window()
    show_window()