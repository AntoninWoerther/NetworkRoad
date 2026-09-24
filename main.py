import heapq
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button, Slider, TextBox

from node import Node, NodeType
from segment import Segment


GRID_SIZE = (24, 12)
FPS = 60
MAX_SEED = 999_999_999

C = {
    "bg": "#061019",
    "panel": "#0B1823",
    "panel_2": "#102536",
    "border": "#17384B",
    "grid": "#173044",
    "grid_bright": "#24495F",
    "road": "#C8D7E2",
    "cyan": "#28D7FF",
    "cyan_2": "#00AEEB",
    "green": "#31E6A1",
    "red": "#FF6078",
    "orange": "#FFB84A",
    "white": "#F4F8FB",
    "muted": "#7890A3",
}


class RoadNetworkApp:
    def __init__(self):
        self.cols, self.rows = GRID_SIZE

        self.route_count = 5
        self.vehicle_count = 10
        self.build_speed = 1.0
        self.traffic_speed = 1.0

        self.paused = False
        self.seed = random.SystemRandom().randint(0, MAX_SEED)
        self.rng = random.Random(self.seed)

        self.nodes = {}
        self.routes = []
        self.segments = []
        self.segment_by_key = {}
        self.forward_graph = defaultdict(list)
        self.node_usage = defaultdict(int)

        self.start_node = None
        self.end_node = None
        self.shortest_path_keys = set()

        self.vehicles = []
        self.build_position = 0.0
        self.build_finished = False
        self.last_revealed_column = -1
        self.frame_counter = 0

        self.fig, self.ax = self.create_window()
        self.create_interface()

        self.generate_network(seed=self.seed, new_endpoints=True)
        self.rebuild_scene()
        self.sync_seed_box()

        self.animation = FuncAnimation(
            self.fig,
            self.update,
            interval=1000 / FPS,
            blit=False,
            cache_frame_data=False,
        )

    # ------------------------------------------------------------------
    # WINDOW / UI
    # ------------------------------------------------------------------

    def create_window(self):
        plt.rcParams["toolbar"] = "None"

        fig = plt.figure(figsize=(15, 8.5), facecolor=C["bg"])

        try:
            fig.canvas.manager.set_window_title("RoadNetwork")
        except Exception:
            pass

        ax = fig.add_axes([0.045, 0.235, 0.705, 0.615])
        ax.set_facecolor(C["bg"])
        ax.set_xlim(-0.9, self.cols - 0.1)
        ax.set_ylim(-0.9, self.rows - 0.1)
        ax.set_aspect("equal")
        ax.axis("off")

        return fig, ax

    def create_interface(self):
        self.fig.text(
            0.05,
            0.955,
            "ROAD NETWORK",
            color=C["white"],
            fontsize=24,
            fontweight="bold",
        )

        self.fig.text(
            0.05,
            0.922,
            "Procedural routing engine  /  strictly forward  /  deterministic seeds",
            color=C["muted"],
            fontsize=10,
        )

        self.header_status = self.fig.text(
            0.05,
            0.882,
            "",
            color=C["cyan"],
            fontsize=9.5,
            family="monospace",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": C["panel"],
                "edgecolor": C["border"],
                "linewidth": 1,
            },
        )

        # ---------- side card ----------
        self.side_ax = self.fig.add_axes([0.775, 0.235, 0.195, 0.615])
        self.side_ax.set_facecolor(C["panel"])
        self.side_ax.set_xlim(0, 1)
        self.side_ax.set_ylim(0, 1)
        self.side_ax.set_xticks([])
        self.side_ax.set_yticks([])

        for spine in self.side_ax.spines.values():
            spine.set_color(C["border"])
            spine.set_linewidth(1.1)

        self.side_ax.text(
            0.09,
            0.94,
            "NETWORK STATUS",
            color=C["white"],
            fontsize=12,
            fontweight="bold",
            va="top",
        )

        self.side_ax.text(
            0.09,
            0.895,
            "LIVE SIMULATION",
            color=C["cyan"],
            fontsize=7.5,
            family="monospace",
            va="top",
        )

        self.side_ax.plot(
            [0.09, 0.91],
            [0.855, 0.855],
            color=C["border"],
            linewidth=1,
        )

        self.stat_labels = {}
        stat_rows = [
            ("state", "State"),
            ("seed", "Seed"),
            ("segments", "Segments"),
            ("routes", "Routes"),
            ("intersections", "Junctions"),
            ("vehicles", "Vehicles"),
        ]

        y = 0.80
        for key, label in stat_rows:
            self.side_ax.text(
                0.09,
                y,
                label.upper(),
                color=C["muted"],
                fontsize=8,
                family="monospace",
                va="center",
            )
            self.stat_labels[key] = self.side_ax.text(
                0.91,
                y,
                "",
                color=C["white"],
                fontsize=9.2,
                family="monospace",
                ha="right",
                va="center",
            )
            y -= 0.064

        self.side_ax.plot(
            [0.09, 0.91],
            [0.385, 0.385],
            color=C["border"],
            linewidth=1,
        )

        self.side_ax.text(
            0.09,
            0.345,
            "LEGEND",
            color=C["white"],
            fontsize=9.5,
            fontweight="bold",
        )

        legend = [
            (C["green"], "Start"),
            (C["red"], "End"),
            (C["orange"], "Junction"),
            (C["road"], "Connection"),
            (C["cyan"], "Shortest path"),
        ]

        y = 0.295
        for color, label in legend:
            self.side_ax.scatter([0.13], [y], s=58, color=color, zorder=2)
            self.side_ax.text(
                0.23,
                y,
                label,
                color=C["white"],
                fontsize=8.8,
                va="center",
            )
            y -= 0.048

        # A dedicated rule card prevents the old legend/text overlap.
        self.side_ax.text(
            0.09,
            0.025,
            "RULE  x → x + 1   •   no backward edge",
            color=C["cyan"],
            fontsize=7.4,
            family="monospace",
            va="bottom",
        )

        # ---------- controls ----------
        self.fig.text(
            0.05,
            0.185,
            "CONTROLS",
            color=C["muted"],
            fontsize=8,
            family="monospace",
        )

        self.sliders = {}
        self.sliders["Routes"] = self.make_slider(
            [0.075, 0.135, 0.205, 0.020], "Routes", 1, 9, self.route_count, 1
        )
        self.sliders["Vehicles"] = self.make_slider(
            [0.075, 0.087, 0.205, 0.020], "Vehicles", 1, 30, self.vehicle_count, 1
        )
        self.sliders["Construction"] = self.make_slider(
            [0.355, 0.135, 0.205, 0.020], "Build", 0.25, 4.0, self.build_speed, None
        )
        self.sliders["Traffic"] = self.make_slider(
            [0.355, 0.087, 0.205, 0.020], "Traffic", 0.25, 4.0, self.traffic_speed, None
        )

        self.sliders["Routes"].on_changed(self.on_route_count)
        self.sliders["Vehicles"].on_changed(self.on_vehicle_count)
        self.sliders["Construction"].on_changed(self.on_speed_change)
        self.sliders["Traffic"].on_changed(self.on_speed_change)

        self.random_button = self.make_button([0.60, 0.132, 0.095, 0.045], "RANDOMIZE")
        self.pause_button = self.make_button([0.705, 0.132, 0.075, 0.045], "PAUSE")
        self.replay_button = self.make_button([0.79, 0.132, 0.075, 0.045], "REPLAY")

        self.new_path_button = self.make_button(
            [0.60, 0.075, 0.18, 0.040], "NEW ROADS · SAME ENDS"
        )

        seed_ax = self.fig.add_axes([0.80, 0.075, 0.105, 0.040])
        seed_ax.set_facecolor(C["panel"])
        self.seed_box = TextBox(
            seed_ax,
            "",
            initial=str(self.seed),
            color=C["panel"],
            hovercolor=C["panel_2"],
        )
        self.seed_box.text_disp.set_color(C["white"])
        self.seed_box.text_disp.set_fontfamily("monospace")
        self.seed_box.text_disp.set_fontsize(9)
        for spine in seed_ax.spines.values():
            spine.set_color(C["border"])

        self.load_seed_button = self.make_button([0.915, 0.075, 0.055, 0.040], "LOAD")

        self.fig.text(
            0.80,
            0.119,
            "SEED",
            color=C["muted"],
            fontsize=7.5,
            family="monospace",
        )

        self.seed_feedback = self.fig.text(
            0.80,
            0.052,
            "Enter a seed to reproduce a network",
            color=C["muted"],
            fontsize=7.2,
        )

        self.random_button.on_clicked(self.randomize)
        self.pause_button.on_clicked(self.toggle_pause)
        self.replay_button.on_clicked(self.replay)
        self.new_path_button.on_clicked(self.new_roads_same_endpoints)
        self.load_seed_button.on_clicked(self.load_seed)
        self.seed_box.on_submit(self.load_seed)

    def make_slider(self, bounds, label, minimum, maximum, initial, step):
        slider_ax = self.fig.add_axes(bounds)
        slider_ax.set_facecolor(C["panel"])

        slider = Slider(
            slider_ax,
            label,
            minimum,
            maximum,
            valinit=initial,
            valstep=step,
            color=C["cyan"],
        )
        slider.label.set_color(C["white"])
        slider.label.set_fontsize(8.5)
        slider.valtext.set_color(C["white"])
        slider.valtext.set_fontsize(8.5)
        return slider

    def make_button(self, bounds, label):
        button_ax = self.fig.add_axes(bounds)
        button_ax.set_zorder(50)

        for spine in button_ax.spines.values():
            spine.set_color(C["border"])

        button = Button(
            button_ax,
            label,
            color=C["panel"],
            hovercolor=C["panel_2"],
        )
        button.label.set_color(C["white"])
        button.label.set_fontsize(8.2)
        return button

    # ------------------------------------------------------------------
    # NETWORK GENERATION
    # ------------------------------------------------------------------

    def make_grid(self):
        self.nodes = {}
        for y in range(self.rows):
            for x in range(self.cols):
                node_id = y * self.cols + x
                self.nodes[(x, y)] = Node(node_id, x, y)

    def generate_network(self, seed=None, new_endpoints=True):
        previous_start_y = self.start_node.y if self.start_node is not None else None
        previous_end_y = self.end_node.y if self.end_node is not None else None

        if seed is not None:
            self.seed = int(seed) % (MAX_SEED + 1)

        self.rng = random.Random(self.seed)
        self.make_grid()

        self.routes = []
        self.segments = []
        self.segment_by_key = {}
        self.forward_graph = defaultdict(list)
        self.node_usage = defaultdict(int)

        if new_endpoints or previous_start_y is None or previous_end_y is None:
            start_y = self.rng.randint(2, self.rows - 3)
            end_y = self.rng.randint(2, self.rows - 3)
        else:
            start_y = previous_start_y
            end_y = previous_end_y

        self.start_node = self.nodes[(0, start_y)]
        self.end_node = self.nodes[(self.cols - 1, end_y)]

        for route_index in range(self.route_count):
            route = self.generate_route(route_index)
            self.routes.append(route)

            for node in route:
                self.node_usage[node] += 1

            for a, b in zip(route, route[1:]):
                self.register_segment(a, b)

        self.classify_nodes()
        self.compute_shortest_path()

    def generate_route(self, route_index):
        """Create a route where every edge is exactly x -> x + 1."""
        route = [self.start_node]
        current = self.start_node
        last_dy = 0

        for next_x in range(1, self.cols):
            remaining = self.cols - 1 - next_x

            if next_x == self.cols - 1:
                next_y = self.end_node.y
            else:
                valid = []

                for dy in (-1, 0, 1):
                    y = current.y + dy
                    if not 0 <= y < self.rows:
                        continue

                    # The destination must still be reachable.
                    if abs(self.end_node.y - y) > remaining:
                        continue

                    candidate = self.nodes[(next_x, y)]

                    # No X-shaped crossings between two diagonal edges.
                    if self.edge_would_cross(current, candidate):
                        continue

                    valid.append((dy, y))

                if not valid:
                    # Reachability-safe fallback. Crossing prevention is relaxed
                    # only if every clean candidate is impossible.
                    for dy in (-1, 0, 1):
                        y = current.y + dy
                        if 0 <= y < self.rows and abs(self.end_node.y - y) <= remaining:
                            valid.append((dy, y))

                scored = []

                for dy, y in valid:
                    candidate = self.nodes[(next_x, y)]

                    reuse_penalty = self.node_usage.get(candidate, 0) * 0.50
                    reversal_penalty = 0.85 if last_dy != 0 and dy == -last_dy else 0.0
                    momentum_bonus = 0.28 if dy == last_dy else 0.0
                    straight_bonus = 0.08 if dy == 0 else 0.0
                    target_pull = abs(self.end_node.y - y) * 0.03

                    # Different routes tend to occupy different vertical lanes.
                    lane = (route_index - (self.route_count - 1) / 2) * 0.035
                    lane_bonus = lane * (y - self.rows / 2)

                    score = (
                        self.rng.random()
                        + momentum_bonus
                        + straight_bonus
                        + lane_bonus
                        - reuse_penalty
                        - reversal_penalty
                        - target_pull
                    )
                    scored.append((score, dy, y))

                scored.sort(key=lambda item: item[0], reverse=True)
                _, last_dy, next_y = scored[0]

            current = self.nodes[(next_x, next_y)]
            route.append(current)

        return route

    def edge_would_cross(self, a, b):
        """Reject diagonal X-crossings that do not meet on a grid node."""
        for segment in self.segments:
            if segment.start.x != a.x:
                continue

            d0 = a.y - segment.start.y
            d1 = b.y - segment.end.y

            if d0 * d1 < 0:
                return True

        return False

    def register_segment(self, a, b):
        key = (a.id, b.id)

        if key not in self.segment_by_key:
            segment = Segment(a, b)
            self.segment_by_key[key] = segment
            self.segments.append(segment)
            self.forward_graph[a].append((b, segment))
            return segment

        return self.segment_by_key[key]

    def classify_nodes(self):
        incoming = defaultdict(int)
        outgoing = defaultdict(int)

        for a, edges in self.forward_graph.items():
            outgoing[a] += len(edges)
            for b, _segment in edges:
                incoming[b] += 1

        for node in self.nodes.values():
            degree = incoming[node] + outgoing[node]

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
        self.shortest_path_keys = set()

        distances = {self.start_node: 0.0}
        previous = {}
        queue = [(0.0, self.start_node.id, self.start_node)]

        while queue:
            distance, _node_id, node = heapq.heappop(queue)

            if distance != distances.get(node):
                continue

            if node == self.end_node:
                break

            for neighbour, segment in self.forward_graph[node]:
                candidate = distance + segment.length

                if candidate < distances.get(neighbour, float("inf")):
                    distances[neighbour] = candidate
                    previous[neighbour] = (node, segment)
                    heapq.heappush(queue, (candidate, neighbour.id, neighbour))

        node = self.end_node
        while node in previous:
            parent, segment = previous[node]
            self.shortest_path_keys.add((segment.start.id, segment.end.id))
            node = parent

    # ------------------------------------------------------------------
    # DRAWING
    # ------------------------------------------------------------------

    def rebuild_scene(self):
        self.ax.clear()
        self.ax.set_facecolor(C["bg"])
        self.ax.set_xlim(-0.9, self.cols - 0.1)
        self.ax.set_ylim(-0.9, self.rows - 0.1)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        # Subtle horizontal guides give the map depth without visual noise.
        for y in range(self.rows):
            self.ax.plot(
                [-0.2, self.cols - 0.8],
                [y, y],
                color=C["grid"],
                linewidth=0.35,
                alpha=0.18,
                zorder=0,
            )

        xs = [node.x for node in self.nodes.values()]
        ys = [node.y for node in self.nodes.values()]
        self.ax.scatter(xs, ys, s=9, color=C["grid_bright"], alpha=0.52, zorder=1)

        self.road_glow = LineCollection(
            [], colors=C["cyan"], linewidths=8.0, alpha=0.075, zorder=2
        )
        self.road_lines = LineCollection(
            [], colors=C["road"], linewidths=2.15, alpha=0.92, zorder=3
        )
        self.path_glow = LineCollection(
            [], colors=C["cyan"], linewidths=4.5, alpha=0.92, zorder=4
        )

        self.ax.add_collection(self.road_glow)
        self.ax.add_collection(self.road_lines)
        self.ax.add_collection(self.path_glow)

        self.start_marker = self.ax.scatter(
            [self.start_node.x], [self.start_node.y],
            s=105, color=C["green"], edgecolors=C["white"], linewidths=1.0, zorder=10
        )
        self.end_marker = self.ax.scatter(
            [self.end_node.x], [self.end_node.y],
            s=105, color=C["red"], edgecolors=C["white"], linewidths=1.0, zorder=10
        )
        self.start_ring = self.ax.scatter(
            [self.start_node.x], [self.start_node.y],
            s=220, facecolors="none", edgecolors=C["green"], linewidths=1.1,
            alpha=0.25, zorder=9
        )
        self.end_ring = self.ax.scatter(
            [self.end_node.x], [self.end_node.y],
            s=220, facecolors="none", edgecolors=C["red"], linewidths=1.1,
            alpha=0.25, zorder=9
        )

        self.ax.text(
            self.start_node.x + 0.25, self.start_node.y + 0.35, "START",
            color=C["green"], fontsize=7.5, fontweight="bold", zorder=11
        )
        self.ax.text(
            self.end_node.x - 0.25, self.end_node.y + 0.35, "END",
            color=C["red"], fontsize=7.5, fontweight="bold", ha="right", zorder=11
        )

        self.intersection_scatter = self.ax.scatter(
            [], [], s=58, color=C["orange"], edgecolors=C["bg"], linewidths=1.2, zorder=8
        )
        self.connection_scatter = self.ax.scatter(
            [], [], s=17, color=C["road"], edgecolors=C["bg"], linewidths=0.7, zorder=7
        )
        self.build_head_scatter = self.ax.scatter(
            [], [], s=35, color=C["cyan"], edgecolors=C["white"], linewidths=0.6, zorder=11
        )
        self.vehicle_glow_scatter = self.ax.scatter(
            [], [], s=150, color=C["cyan"], alpha=0.12, edgecolors="none", zorder=11
        )
        self.vehicle_scatter = self.ax.scatter(
            [], [], s=42, color=C["white"], edgecolors=C["cyan"], linewidths=1.5, zorder=12
        )

        self.build_position = 0.0
        self.build_finished = False
        self.last_revealed_column = -1
        self.make_vehicles()

        self.update_build_drawing()
        self.update_status_text()
        self.fig.canvas.draw_idle()

    def update_build_drawing(self):
        roads = []
        shortest = []
        heads = []

        for segment in self.segments:
            progress = max(0.0, min(1.0, self.build_position - segment.start.x))

            if progress <= 0.0:
                continue

            x, y = segment.point_at(progress)
            line = [(segment.start.x, segment.start.y), (x, y)]
            roads.append(line)

            key = (segment.start.id, segment.end.id)
            if key in self.shortest_path_keys:
                shortest.append(line)

            if 0.0 < progress < 1.0:
                heads.append((x, y))

        self.road_glow.set_segments(roads)
        self.road_lines.set_segments(roads)
        self.path_glow.set_segments(shortest)

        if heads:
            self.build_head_scatter.set_offsets(heads)
        else:
            self.build_head_scatter.set_offsets(np.empty((0, 2)))

        reveal_column = min(self.cols - 1, int(self.build_position + 0.02))
        if reveal_column != self.last_revealed_column:
            self.last_revealed_column = reveal_column
            self.update_visible_nodes(reveal_column)

    def update_visible_nodes(self, reveal_column):
        junctions = []
        connections = []

        for node in self.nodes.values():
            if node.x > reveal_column:
                continue

            if node.type == NodeType.Intersection:
                junctions.append((node.x, node.y))
            elif node.type == NodeType.Connection:
                connections.append((node.x, node.y))

        self.intersection_scatter.set_offsets(
            junctions if junctions else np.empty((0, 2))
        )
        self.connection_scatter.set_offsets(
            connections if connections else np.empty((0, 2))
        )

    # ------------------------------------------------------------------
    # VEHICLES
    # ------------------------------------------------------------------

    def make_vehicles(self):
        self.vehicles = []
        first_edges = self.forward_graph[self.start_node]

        if not first_edges:
            return

        for index in range(self.vehicle_count):
            next_node, segment = self.rng.choice(first_edges)
            self.vehicles.append(
                {
                    "from": self.start_node,
                    "to": next_node,
                    "segment": segment,
                    "progress": (index / max(self.vehicle_count, 1)) * 0.9,
                    "speed": self.rng.uniform(0.010, 0.017),
                }
            )

    def restart_vehicle(self, vehicle):
        choices = self.forward_graph[self.start_node]
        if not choices:
            return False

        next_node, segment = self.rng.choice(choices)
        vehicle["from"] = self.start_node
        vehicle["to"] = next_node
        vehicle["segment"] = segment
        vehicle["progress"] = 0.0
        return True

    def update_vehicles(self):
        if not self.build_finished:
            self.vehicle_scatter.set_offsets(np.empty((0, 2)))
            self.vehicle_glow_scatter.set_offsets(np.empty((0, 2)))
            return

        positions = []

        for vehicle in self.vehicles:
            vehicle["progress"] += vehicle["speed"] * self.traffic_speed

            while vehicle["progress"] >= 1.0:
                overflow = vehicle["progress"] - 1.0
                current = vehicle["to"]

                if current == self.end_node:
                    if not self.restart_vehicle(vehicle):
                        break
                else:
                    choices = self.forward_graph[current]
                    if not choices:
                        if not self.restart_vehicle(vehicle):
                            break
                    else:
                        next_node, segment = self.rng.choice(choices)
                        vehicle["from"] = current
                        vehicle["to"] = next_node
                        vehicle["segment"] = segment
                        vehicle["progress"] = 0.0

                vehicle["progress"] += overflow

            positions.append(vehicle["segment"].point_at(vehicle["progress"]))

        offsets = positions if positions else np.empty((0, 2))
        self.vehicle_glow_scatter.set_offsets(offsets)
        self.vehicle_scatter.set_offsets(offsets)

    # ------------------------------------------------------------------
    # STATE / CALLBACKS
    # ------------------------------------------------------------------

    def update_status_text(self):
        junctions = sum(
            node.type == NodeType.Intersection for node in self.nodes.values()
        )

        if self.paused:
            state = "PAUSED"
        elif self.build_finished:
            state = "TRAFFIC"
        else:
            percent = min(100, int(self.build_position / (self.cols - 1) * 100))
            state = f"BUILD {percent:02d}%"

        self.header_status.set_text(
            f"SEED {self.seed:09d}   •   {self.route_count} ROUTES   •   "
            f"{len(self.segments)} LINKS"
        )

        self.stat_labels["state"].set_text(state)
        self.stat_labels["seed"].set_text(f"{self.seed:09d}")
        self.stat_labels["segments"].set_text(str(len(self.segments)))
        self.stat_labels["routes"].set_text(str(self.route_count))
        self.stat_labels["intersections"].set_text(str(junctions))
        self.stat_labels["vehicles"].set_text(str(self.vehicle_count))

    def update(self, _frame):
        self.frame_counter += 1

        # Small pulse on the endpoint rings; cheap but makes the UI feel alive.
        pulse = 220 + 18 * np.sin(self.frame_counter * 0.08)
        self.start_ring.set_sizes([pulse])
        self.end_ring.set_sizes([pulse])

        if self.paused:
            return

        if not self.build_finished:
            self.build_position += 0.095 * self.build_speed

            if self.build_position >= self.cols - 1:
                self.build_position = self.cols - 1
                self.build_finished = True
                self.build_head_scatter.set_offsets(np.empty((0, 2)))

            self.update_build_drawing()

            # Updating text less often avoids unnecessary text layout work.
            if self.frame_counter % 4 == 0 or self.build_finished:
                self.update_status_text()
        else:
            self.update_vehicles()

    def on_route_count(self, value):
        new_count = int(value)
        if new_count == self.route_count:
            return

        self.route_count = new_count
        # Same seed + new route count = deterministic variation.
        self.generate_network(seed=self.seed, new_endpoints=True)
        self.rebuild_scene()

    def on_vehicle_count(self, value):
        self.vehicle_count = int(value)
        self.make_vehicles()
        self.update_status_text()
        self.fig.canvas.draw_idle()

    def on_speed_change(self, _value):
        self.build_speed = float(self.sliders["Construction"].val)
        self.traffic_speed = float(self.sliders["Traffic"].val)

    def randomize(self, _event):
        self.seed = random.SystemRandom().randint(0, MAX_SEED)
        self.paused = False
        self.pause_button.label.set_text("PAUSE")

        self.generate_network(seed=self.seed, new_endpoints=True)
        self.rebuild_scene()
        self.sync_seed_box()
        self.seed_feedback.set_text("New random seed generated")
        self.seed_feedback.set_color(C["green"])

    def new_roads_same_endpoints(self, _event):
        new_seed = random.SystemRandom().randint(0, MAX_SEED)
        self.paused = False
        self.pause_button.label.set_text("PAUSE")

        self.generate_network(seed=new_seed, new_endpoints=False)
        self.rebuild_scene()
        self.sync_seed_box()
        self.seed_feedback.set_text("New roads · endpoints preserved")
        self.seed_feedback.set_color(C["cyan"])

    def replay(self, _event):
        self.paused = False
        self.pause_button.label.set_text("PAUSE")
        self.build_position = 0.0
        self.build_finished = False
        self.last_revealed_column = -1
        self.make_vehicles()
        self.update_build_drawing()
        self.update_status_text()
        self.vehicle_scatter.set_offsets(np.empty((0, 2)))
        self.vehicle_glow_scatter.set_offsets(np.empty((0, 2)))
        self.fig.canvas.draw_idle()

    def toggle_pause(self, _event):
        self.paused = not self.paused
        self.pause_button.label.set_text("RESUME" if self.paused else "PAUSE")
        self.update_status_text()
        self.fig.canvas.draw_idle()

    def load_seed(self, _event=None):
        raw = self.seed_box.text.strip()

        try:
            seed = int(raw)
            if seed < 0:
                raise ValueError
        except ValueError:
            self.seed_feedback.set_text("Invalid seed · use a positive integer")
            self.seed_feedback.set_color(C["red"])
            return

        self.seed = seed % (MAX_SEED + 1)
        self.paused = False
        self.pause_button.label.set_text("PAUSE")

        self.generate_network(seed=self.seed, new_endpoints=True)
        self.rebuild_scene()
        self.sync_seed_box()
        self.seed_feedback.set_text("Seed loaded · network reproduced")
        self.seed_feedback.set_color(C["green"])

    def sync_seed_box(self):
        if not hasattr(self, "seed_box"):
            return

        previous = self.seed_box.eventson
        self.seed_box.eventson = False
        self.seed_box.set_val(str(self.seed))
        self.seed_box.eventson = previous

    def show(self):
        plt.show()


if __name__ == "__main__":
    RoadNetworkApp().show()
