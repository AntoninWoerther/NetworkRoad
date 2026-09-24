import heapq
import random
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

from node import Node, NodeType
from segment import Segment


GRID_SIZE = (24, 12)

COLORS = {
    "background": "#081019",
    "panel": "#101B27",
    "grid": "#243545",
    "road": "#D8E3EC",
    "glow": "#2CCBFF",
    "start": "#35E59A",
    "end": "#FF5C74",
    "intersection": "#FFB84A",
    "vehicle": "#FFFFFF",
    "text": "#EEF7FF",
    "muted": "#8295A7",
}


class RoadNetworkApp:
    def __init__(self):
        self.cols, self.rows = GRID_SIZE

        self.route_count = 5
        self.intersection_target = 5
        self.vehicle_count = 10
        self.build_speed = 1.0
        self.traffic_speed = 1.0

        self.paused = False
        self.seed = random.randint(0, 99999)
        self.rng = random.Random(self.seed)

        self.nodes = {}
        self.segments = []
        self.graph = defaultdict(list)

        self.start_node = None
        self.end_node = None
        self.path_segment_keys = set()

        self.segment_artists = []
        self.path_artists = []
        self.vehicle_artists = []
        self.vehicles = []

        self.build_index = 0
        self.build_progress = 0.0

        self.fig, self.ax = self.create_window()
        self.create_ui()

        self.generate()
        self.redraw()

        self.animation = FuncAnimation(
            self.fig,
            self.update,
            interval=16,
            blit=False,
            cache_frame_data=False,
        )

    def create_window(self):
        plt.rcParams["toolbar"] = "None"

        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor(COLORS["background"])
        ax.set_facecolor(COLORS["background"])

        try:
            fig.canvas.manager.set_window_title("RoadNetwork")
        except Exception:
            pass

        fig.subplots_adjust(left=0.05, right=0.78, bottom=0.20, top=0.88)

        ax.set_xlim(-1, self.cols)
        ax.set_ylim(-1, self.rows)
        ax.set_aspect("equal")
        ax.axis("off")

        return fig, ax

    def create_ui(self):
        self.fig.text(
            0.05, 0.95,
            "ROAD NETWORK",
            color=COLORS["text"],
            fontsize=22,
            fontweight="bold",
        )

        self.fig.text(
            0.05, 0.918,
            "Réseau procédural • génération animée • trafic autonome",
            color=COLORS["muted"],
            fontsize=10,
        )

        self.status = self.fig.text(
            0.05, 0.885, "",
            color=COLORS["glow"],
            fontsize=10,
            family="monospace",
        )

        slider_data = [
            ("Routes", 2, 10, self.route_count, 1),
            ("Intersections", 0, 12, self.intersection_target, 1),
            ("Véhicules", 1, 30, self.vehicle_count, 1),
            ("Construction", 0.25, 3.0, self.build_speed, None),
            ("Trafic", 0.25, 3.0, self.traffic_speed, None),
        ]

        self.sliders = {}

        y = 0.145
        for label, minimum, maximum, value, step in slider_data:
            slider_ax = self.fig.add_axes([0.08, y, 0.25, 0.022])
            slider_ax.set_facecolor(COLORS["panel"])

            slider = Slider(
                slider_ax,
                label,
                minimum,
                maximum,
                valinit=value,
                valstep=step,
                color=COLORS["glow"],
            )

            slider.label.set_color(COLORS["text"])
            slider.valtext.set_color(COLORS["text"])

            self.sliders[label] = slider
            y -= 0.031

        self.sliders["Routes"].on_changed(self.on_network_slider)
        self.sliders["Intersections"].on_changed(self.on_network_slider)
        self.sliders["Véhicules"].on_changed(self.on_vehicle_slider)
        self.sliders["Construction"].on_changed(self.on_speed_slider)
        self.sliders["Trafic"].on_changed(self.on_speed_slider)

        random_ax = self.fig.add_axes([0.37, 0.105, 0.12, 0.052])
        pause_ax = self.fig.add_axes([0.50, 0.105, 0.10, 0.052])
        path_ax = self.fig.add_axes([0.61, 0.105, 0.12, 0.052])

        self.random_button = Button(
            random_ax, "RANDOMIZE",
            color=COLORS["panel"],
            hovercolor="#1A2A3A",
        )
        self.pause_button = Button(
            pause_ax, "PAUSE",
            color=COLORS["panel"],
            hovercolor="#1A2A3A",
        )
        self.path_button = Button(
            path_ax, "NEW PATH",
            color=COLORS["panel"],
            hovercolor="#1A2A3A",
        )

        for button in (
            self.random_button,
            self.pause_button,
            self.path_button,
        ):
            button.label.set_color(COLORS["text"])

        self.random_button.on_clicked(self.randomize)
        self.pause_button.on_clicked(self.toggle_pause)
        self.path_button.on_clicked(self.change_endpoints)

        self.fig.text(
            0.81, 0.72,
            "NETWORK STATUS",
            color=COLORS["text"],
            fontsize=12,
            fontweight="bold",
        )

        self.stats = self.fig.text(
            0.81, 0.67, "",
            color=COLORS["muted"],
            fontsize=10,
            family="monospace",
            linespacing=1.6,
        )

        legend = [
            (COLORS["start"], "Start"),
            (COLORS["end"], "End"),
            (COLORS["intersection"], "Intersection"),
            (COLORS["road"], "Connection"),
            (COLORS["glow"], "Shortest path"),
        ]

        y = 0.50
        for color, name in legend:
            self.fig.text(0.815, y, "●", color=color, fontsize=15, va="center")
            self.fig.text(
                0.845, y, name,
                color=COLORS["text"],
                fontsize=10,
                va="center",
            )
            y -= 0.045

    def make_grid(self):
        self.nodes = {}

        for y in range(self.rows):
            for x in range(self.cols):
                node_id = y * self.cols + x
                self.nodes[(x, y)] = Node(node_id, x, y)

    def forward_neighbours(self, node):
        """Only nodes in the next column are valid: never go backwards."""
        result = []

        next_x = node.x + 1
        if next_x >= self.cols:
            return result

        for dy in (-1, 0, 1):
            next_y = node.y + dy
            key = (next_x, next_y)

            if key in self.nodes:
                result.append(self.nodes[key])

        return result

    def has_segment(self, a, b):
        key = tuple(sorted((a.id, b.id)))
        return any(segment.key == key for segment in self.segments)

    def connect(self, a, b):
        if a == b or self.has_segment(a, b):
            return False

        segment = Segment(
            a,
            b,
            bend=self.rng.uniform(-0.16, 0.16),
        )

        self.segments.append(segment)
        self.graph[a].append((b, segment))
        self.graph[b].append((a, segment))
        return True

    def generate(self):
        """
        Generate one or several routes from a fixed point on the LEFT
        to a fixed point on the RIGHT.

        Absolute rule:
        every segment goes from column x to column x + 1.
        A route can go up, straight, or down, but NEVER backwards.
        """
        self.seed = random.randint(0, 99999)
        self.rng = random.Random(self.seed)

        self.make_grid()
        self.segments = []
        self.graph = defaultdict(list)

        start_y = self.rng.randint(2, self.rows - 3)
        end_y = self.rng.randint(2, self.rows - 3)

        self.start_node = self.nodes[(0, start_y)]
        self.end_node = self.nodes[(self.cols - 1, end_y)]

        # Each route independently travels from left to right.
        # Routes may cross/share nodes, creating intersections naturally.
        for route_index in range(self.route_count):
            current = self.start_node

            for x in range(self.cols - 1):
                remaining_after_move = self.cols - 2 - x
                candidates = []

                for candidate in self.forward_neighbours(current):
                    # Keep only moves that still make the final Y reachable.
                    if abs(self.end_node.y - candidate.y) <= remaining_after_move:
                        candidates.append(candidate)

                if not candidates:
                    # Defensive fallback; normally unreachable because of the
                    # reachability rule above.
                    target_y = max(
                        0,
                        min(
                            self.rows - 1,
                            current.y + (
                                1 if self.end_node.y > current.y
                                else -1 if self.end_node.y < current.y
                                else 0
                            ),
                        ),
                    )
                    candidates = [self.nodes[(current.x + 1, target_y)]]

                # Prefer randomness, with a light attraction to the end point.
                # Different route indexes receive a small vertical personality
                # so they do not all collapse onto the same line.
                scored = []
                route_bias = ((route_index % 3) - 1) * 0.16

                for candidate in candidates:
                    distance_to_end = abs(self.end_node.y - candidate.y)
                    straight_bonus = 0.20 if candidate.y == current.y else 0.0
                    divergence = route_bias * (candidate.y - current.y)

                    score = (
                        self.rng.random()
                        - distance_to_end * 0.07
                        + straight_bonus
                        + divergence
                    )
                    scored.append((score, candidate))

                scored.sort(key=lambda item: item[0], reverse=True)
                nxt = scored[0][1]

                self.connect(current, nxt)
                current = nxt

        self.classify_nodes()
        self.compute_shortest_path()

    def classify_nodes(self):
        for node in self.nodes.values():
            degree = len(self.graph[node])

            if node == self.start_node:
                node.type = NodeType.Start
            elif node == self.end_node:
                node.type = NodeType.End
            elif degree >= 3:
                node.type = NodeType.Intersection
            elif degree > 0:
                node.type = NodeType.Connection
            else:
                node.type = NodeType.Unused

    def compute_shortest_path(self):
        self.path_segment_keys = set()

        if self.start_node is None or self.end_node is None:
            return

        distances = {self.start_node: 0.0}
        previous = {}
        queue = [(0.0, self.start_node.id, self.start_node)]

        while queue:
            distance, _, node = heapq.heappop(queue)

            if distance != distances.get(node):
                continue

            if node == self.end_node:
                break

            for neighbour, segment in self.graph[node]:
                candidate = distance + segment.length

                if candidate < distances.get(neighbour, float("inf")):
                    distances[neighbour] = candidate
                    previous[neighbour] = (node, segment)
                    heapq.heappush(
                        queue,
                        (candidate, neighbour.id, neighbour),
                    )

        node = self.end_node

        while node in previous:
            parent, segment = previous[node]
            self.path_segment_keys.add(segment.key)
            node = parent

    def redraw(self):
        self.ax.clear()
        self.ax.set_facecolor(COLORS["background"])
        self.ax.set_xlim(-1, self.cols)
        self.ax.set_ylim(-1, self.rows)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        # Background point map.
        xs = [node.x for node in self.nodes.values()]
        ys = [node.y for node in self.nodes.values()]

        self.ax.scatter(
            xs, ys,
            s=12,
            color=COLORS["grid"],
            alpha=0.75,
            zorder=1,
        )

        self.segment_artists = []
        self.path_artists = []

        for segment in self.segments:
            glow, = self.ax.plot(
                [], [],
                color=COLORS["glow"],
                linewidth=7,
                alpha=0.10,
                solid_capstyle="round",
                zorder=2,
            )

            road, = self.ax.plot(
                [], [],
                color=COLORS["road"],
                linewidth=2.4,
                alpha=0.95,
                solid_capstyle="round",
                zorder=3,
            )

            self.segment_artists.append((glow, road))

            if segment.key in self.path_segment_keys:
                path, = self.ax.plot(
                    [], [],
                    color=COLORS["glow"],
                    linewidth=4.8,
                    alpha=0.95,
                    solid_capstyle="round",
                    zorder=4,
                )
            else:
                path = None

            self.path_artists.append(path)

        self.node_scatter = self.ax.scatter([], [], s=1)

        self.vehicle_scatter = self.ax.scatter(
            [], [],
            s=52,
            color=COLORS["vehicle"],
            edgecolors=COLORS["glow"],
            linewidths=1.8,
            zorder=8,
        )

        self.build_index = 0
        self.build_progress = 0.0

        self.make_vehicles()
        self.update_stats()

    def make_vehicles(self):
        self.vehicles = []

        if not self.graph[self.start_node]:
            return

        for _ in range(self.vehicle_count):
            choices = [
                item
                for item in self.graph[self.start_node]
                if item[0].x > self.start_node.x
            ]

            if not choices:
                return

            nxt, segment = self.rng.choice(choices)

            self.vehicles.append({
                "from": self.start_node,
                "to": nxt,
                "segment": segment,
                "progress": self.rng.random() * 0.8,
                "speed": self.rng.uniform(0.006, 0.015),
            })

    def draw_nodes(self):
        xs = []
        ys = []
        colors = []
        sizes = []

        for node in self.nodes.values():
            if node.type == NodeType.Unused:
                continue

            xs.append(node.x)
            ys.append(node.y)

            if node.type == NodeType.Start:
                colors.append(COLORS["start"])
                sizes.append(90)
            elif node.type == NodeType.End:
                colors.append(COLORS["end"])
                sizes.append(90)
            elif node.type == NodeType.Intersection:
                colors.append(COLORS["intersection"])
                sizes.append(55)
            else:
                colors.append(COLORS["road"])
                sizes.append(26)

        self.node_scatter.remove()
        self.node_scatter = self.ax.scatter(
            xs, ys,
            s=sizes,
            c=colors,
            edgecolors=COLORS["background"],
            linewidths=1.0,
            zorder=6,
        )

    def update_build(self):
        if self.build_index >= len(self.segments):
            return

        self.build_progress += 0.035 * self.build_speed

        segment = self.segments[self.build_index]
        progress = min(self.build_progress, 1.0)
        xs, ys = segment.sample(28, progress)

        glow, road = self.segment_artists[self.build_index]
        glow.set_data(xs, ys)
        road.set_data(xs, ys)

        path = self.path_artists[self.build_index]
        if path is not None:
            path.set_data(xs, ys)

        if self.build_progress >= 1.0:
            self.build_index += 1
            self.build_progress = 0.0

            if self.build_index >= len(self.segments):
                self.draw_nodes()

    def update_vehicles(self):
        if self.build_index < len(self.segments):
            self.vehicle_scatter.set_offsets([])
            return

        positions = []

        for vehicle in self.vehicles:
            vehicle["progress"] += vehicle["speed"] * self.traffic_speed

            if vehicle["progress"] >= 1.0:
                current = vehicle["to"]

                # Once the end is reached, restart from the left.
                if current == self.end_node:
                    current = self.start_node

                choices = [
                    item
                    for item in self.graph[current]
                    if item[0].x > current.x
                ]

                if not choices:
                    current = self.start_node
                    choices = [
                        item
                        for item in self.graph[current]
                        if item[0].x > current.x
                    ]

                if not choices:
                    continue

                nxt, segment = self.rng.choice(choices)

                vehicle["from"] = current
                vehicle["to"] = nxt
                vehicle["segment"] = segment
                vehicle["progress"] = 0.0

            segment = vehicle["segment"]

            # Generated segments are always left -> right.
            if segment.start == vehicle["from"]:
                t = vehicle["progress"]
            else:
                t = 1.0 - vehicle["progress"]

            positions.append(segment.point_at(t))

        self.vehicle_scatter.set_offsets(positions)

    def update_stats(self):
        intersections = sum(
            node.type == NodeType.Intersection
            for node in self.nodes.values()
        )

        self.status.set_text(
            f"seed {self.seed:05d}  •  "
            f"routes {self.route_count}  •  "
            f"intersections {intersections}"
        )

        self.stats.set_text(
            f"Segments      {len(self.segments):>3}\n"
            f"Intersections {intersections:>3}\n"
            f"Véhicules     {self.vehicle_count:>3}\n"
            f"Grille       {self.cols}×{self.rows}"
        )

    def update(self, _frame):
        if self.paused:
            return

        self.update_build()
        self.update_vehicles()

    def on_network_slider(self, _value):
        self.route_count = int(self.sliders["Routes"].val)
        self.intersection_target = int(self.sliders["Intersections"].val)

    def on_vehicle_slider(self, _value):
        self.vehicle_count = int(self.sliders["Véhicules"].val)

        if self.build_index >= len(self.segments):
            self.make_vehicles()

        self.update_stats()

    def on_speed_slider(self, _value):
        self.build_speed = float(self.sliders["Construction"].val)
        self.traffic_speed = float(self.sliders["Trafic"].val)

    def randomize(self, _event):
        self.route_count = int(self.sliders["Routes"].val)
        self.intersection_target = int(self.sliders["Intersections"].val)
        self.vehicle_count = int(self.sliders["Véhicules"].val)

        self.generate()
        self.redraw()

    def toggle_pause(self, _event):
        self.paused = not self.paused
        self.pause_button.label.set_text("PLAY" if self.paused else "PAUSE")

    def change_endpoints(self, _event):
        # A new path means new LEFT and RIGHT endpoints,
        # then a complete forward-only regeneration.
        self.generate()
        self.redraw()

    def show(self):
        plt.show()


if __name__ == "__main__":
    app = RoadNetworkApp()
    app.show()
