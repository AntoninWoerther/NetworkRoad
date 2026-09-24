import heapq
import random
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

from node import Node, NodeType
from segment import Segment


GRID_SIZE = (24, 12)
FPS = 60

C = {
    "bg": "#071018",
    "panel": "#0E1A26",
    "panel_2": "#122333",
    "grid": "#1D3445",
    "road": "#BFD0DC",
    "road_dim": "#6F8495",
    "cyan": "#31D6FF",
    "cyan_soft": "#1FA7CA",
    "green": "#36E6A0",
    "red": "#FF6179",
    "orange": "#FFB648",
    "white": "#F1F7FB",
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
        self.seed = random.randint(0, 99999)
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

        self.segment_artists = {}
        self.vehicles = []
        self.build_position = 0.0
        self.build_finished = False

        self.fig, self.ax = self.create_window()
        self.create_interface()

        self.generate_network(new_endpoints=True)
        self.rebuild_scene()

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

        ax = fig.add_axes([0.05, 0.22, 0.70, 0.64])
        ax.set_facecolor(C["bg"])
        ax.set_xlim(-0.8, self.cols - 0.2)
        ax.set_ylim(-0.8, self.rows - 0.2)
        ax.set_aspect("equal")
        ax.axis("off")

        return fig, ax

    def create_interface(self):
        self.fig.text(
            0.05,
            0.95,
            "ROAD NETWORK",
            color=C["white"],
            fontsize=24,
            fontweight="bold",
        )

        self.fig.text(
            0.05,
            0.915,
            "Routes procédurales strictement gauche → droite",
            color=C["muted"],
            fontsize=10.5,
        )

        self.header_status = self.fig.text(
            0.05,
            0.882,
            "",
            color=C["cyan"],
            fontsize=10,
            family="monospace",
        )

        # Dedicated side panel: no figure-text collisions.
        self.side_ax = self.fig.add_axes([0.785, 0.22, 0.185, 0.64])
        self.side_ax.set_facecolor(C["panel"])
        self.side_ax.set_xlim(0, 1)
        self.side_ax.set_ylim(0, 1)
        self.side_ax.set_xticks([])
        self.side_ax.set_yticks([])

        for spine in self.side_ax.spines.values():
            spine.set_color("#1B3345")
            spine.set_linewidth(1.2)

        self.side_ax.text(
            0.10,
            0.93,
            "NETWORK STATUS",
            color=C["white"],
            fontsize=12,
            fontweight="bold",
            va="top",
        )

        self.side_ax.plot(
            [0.10, 0.90],
            [0.875, 0.875],
            color="#203A4D",
            linewidth=1,
        )

        self.stat_labels = {}
        stat_rows = [
            ("state", "State"),
            ("segments", "Segments"),
            ("routes", "Routes"),
            ("intersections", "Intersections"),
            ("vehicles", "Vehicles"),
            ("grid", "Grid"),
        ]

        y = 0.81
        for key, label in stat_rows:
            self.side_ax.text(
                0.10,
                y,
                label.upper(),
                color=C["muted"],
                fontsize=8.5,
                family="monospace",
                va="center",
            )
            self.stat_labels[key] = self.side_ax.text(
                0.90,
                y,
                "",
                color=C["white"],
                fontsize=9.5,
                family="monospace",
                ha="right",
                va="center",
            )
            y -= 0.07

        self.side_ax.text(
            0.10,
            0.37,
            "LEGEND",
            color=C["white"],
            fontsize=10,
            fontweight="bold",
        )

        legend = [
            (C["green"], "Start"),
            (C["red"], "End"),
            (C["orange"], "Intersection"),
            (C["road"], "Connection"),
            (C["cyan"], "Shortest path"),
        ]

        y = 0.31
        for color, label in legend:
            self.side_ax.scatter([0.13], [y], s=65, color=color, zorder=2)
            self.side_ax.text(
                0.23,
                y,
                label,
                color=C["white"],
                fontsize=9,
                va="center",
            )
            y -= 0.055

        self.side_ax.text(
            0.10,
            0.055,
            "Every edge advances exactly\none column to the right.",
            color=C["muted"],
            fontsize=8.5,
            linespacing=1.4,
            va="bottom",
        )

        # Controls use their own axes and stay above the plot.
        self.sliders = {}

        self.sliders["Routes"] = self.make_slider(
            [0.075, 0.135, 0.22, 0.024],
            "Routes",
            1,
            9,
            self.route_count,
            1,
        )

        self.sliders["Vehicles"] = self.make_slider(
            [0.075, 0.082, 0.22, 0.024],
            "Vehicles",
            1,
            30,
            self.vehicle_count,
            1,
        )

        self.sliders["Construction"] = self.make_slider(
            [0.385, 0.135, 0.22, 0.024],
            "Construction",
            0.25,
            4.0,
            self.build_speed,
            None,
        )

        self.sliders["Traffic"] = self.make_slider(
            [0.385, 0.082, 0.22, 0.024],
            "Traffic",
            0.25,
            4.0,
            self.traffic_speed,
            None,
        )

        self.sliders["Routes"].on_changed(self.on_route_count)
        self.sliders["Vehicles"].on_changed(self.on_vehicle_count)
        self.sliders["Construction"].on_changed(self.on_speed_change)
        self.sliders["Traffic"].on_changed(self.on_speed_change)

        self.random_button = self.make_button(
            [0.67, 0.115, 0.09, 0.052],
            "RANDOMIZE",
        )
        self.pause_button = self.make_button(
            [0.775, 0.115, 0.08, 0.052],
            "PAUSE",
        )
        self.replay_button = self.make_button(
            [0.87, 0.115, 0.08, 0.052],
            "REPLAY",
        )

        self.new_path_button = self.make_button(
            [0.775, 0.055, 0.175, 0.042],
            "NEW ROADS · SAME ENDS",
        )

        self.random_button.on_clicked(self.randomize)
        self.pause_button.on_clicked(self.toggle_pause)
        self.replay_button.on_clicked(self.replay)
        self.new_path_button.on_clicked(self.new_roads_same_endpoints)

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
        slider.valtext.set_color(C["white"])
        return slider

    def make_button(self, bounds, label):
        button_ax = self.fig.add_axes(bounds)
        button_ax.set_zorder(50)

        button = Button(
            button_ax,
            label,
            color=C["panel"],
            hovercolor=C["panel_2"],
        )
        button.label.set_color(C["white"])
        button.label.set_fontsize(9)
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

    def generate_network(self, new_endpoints=True):
        self.seed = random.randint(0, 99999)
        self.rng = random.Random(self.seed)

        self.make_grid()
        self.routes = []
        self.segments = []
        self.segment_by_key = {}
        self.forward_graph = defaultdict(list)
        self.node_usage = defaultdict(int)

        if new_endpoints or self.start_node is None or self.end_node is None:
            start_y = self.rng.randint(2, self.rows - 3)
            end_y = self.rng.randint(2, self.rows - 3)
        else:
            start_y = self.start_node.y
            end_y = self.end_node.y

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
        self.update_status_text()

    def generate_route(self, route_index):
        """
        Hard constraint:
        x always increases by exactly 1.

        At every column the route may only choose:
        up-right, right, or down-right.
        It can therefore NEVER go backwards.
        """
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

                    # From this candidate, the destination must remain reachable
                    # with at most one vertical step per remaining column.
                    if abs(self.end_node.y - y) > remaining:
                        continue

                    valid.append((dy, y))

                if not valid:
                    direction = (
                        1
                        if self.end_node.y > current.y
                        else -1
                        if self.end_node.y < current.y
                        else 0
                    )
                    next_y = current.y + direction
                    next_y = max(0, min(self.rows - 1, next_y))
                    last_dy = direction
                else:
                    scored = []

                    for dy, y in valid:
                        candidate = self.nodes[(next_x, y)]

                        # Randomness is dominant, but duplicated paths are
                        # discouraged and violent zig-zags are slightly penalised.
                        reuse_penalty = self.node_usage.get(candidate, 0) * 0.55
                        turn_penalty = 0.32 if last_dy != 0 and dy == -last_dy else 0.0
                        straight_bonus = 0.12 if dy == last_dy else 0.0
                        target_pull = abs(self.end_node.y - y) * 0.035

                        # Give each route a tiny personality so parallel roads
                        # spread naturally instead of perfectly overlapping.
                        lane_bias = ((route_index % 3) - 1) * dy * 0.06

                        score = (
                            self.rng.random()
                            + straight_bonus
                            + lane_bias
                            - reuse_penalty
                            - turn_penalty
                            - target_pull
                        )

                        scored.append((score, dy, y))

                    scored.sort(key=lambda item: item[0], reverse=True)
                    _, last_dy, next_y = scored[0]

            current = self.nodes[(next_x, next_y)]
            route.append(current)

        return route

    def register_segment(self, a, b):
        # Generation only calls this with b.x == a.x + 1.
        key = (a.id, b.id)

        if key not in self.segment_by_key:
            segment = Segment(
                a,
                b,
                bend=self.rng.uniform(-0.12, 0.12),
            )
            self.segment_by_key[key] = segment
            self.segments.append(segment)
            self.forward_graph[a].append((b, segment))
        else:
            segment = self.segment_by_key[key]

        return segment

    def classify_nodes(self):
        incoming = defaultdict(int)
        outgoing = defaultdict(int)

        for a, edges in self.forward_graph.items():
            outgoing[a] += len(edges)
            for b, _segment in edges:
                incoming[b] += 1

        for node in self.nodes.values():
            if node == self.start_node:
                node.type = NodeType.Start
            elif node == self.end_node:
                node.type = NodeType.End
            elif incoming[node] + outgoing[node] >= 3:
                node.type = NodeType.Intersection
            elif incoming[node] + outgoing[node] > 0:
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
                    heapq.heappush(
                        queue,
                        (candidate, neighbour.id, neighbour),
                    )

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
        self.ax.set_xlim(-0.8, self.cols - 0.2)
        self.ax.set_ylim(-0.8, self.rows - 0.2)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        xs = [node.x for node in self.nodes.values()]
        ys = [node.y for node in self.nodes.values()]

        self.ax.scatter(
            xs,
            ys,
            s=11,
            color=C["grid"],
            alpha=0.72,
            zorder=1,
        )

        # Destination markers are always visible.
        self.ax.scatter(
            [self.start_node.x],
            [self.start_node.y],
            s=145,
            color=C["green"],
            edgecolors=C["bg"],
            linewidths=2,
            zorder=10,
        )

        self.ax.scatter(
            [self.end_node.x],
            [self.end_node.y],
            s=145,
            color=C["red"],
            edgecolors=C["bg"],
            linewidths=2,
            zorder=10,
        )

        self.ax.text(
            self.start_node.x + 0.25,
            self.start_node.y + 0.32,
            "START",
            color=C["green"],
            fontsize=8,
            fontweight="bold",
            zorder=11,
        )

        self.ax.text(
            self.end_node.x - 0.25,
            self.end_node.y + 0.32,
            "END",
            color=C["red"],
            fontsize=8,
            fontweight="bold",
            ha="right",
            zorder=11,
        )

        self.segment_artists = {}

        for segment in self.segments:
            key = (segment.start.id, segment.end.id)

            glow, = self.ax.plot(
                [],
                [],
                color=C["cyan"],
                linewidth=8,
                alpha=0.09,
                solid_capstyle="round",
                zorder=2,
            )

            road, = self.ax.plot(
                [],
                [],
                color=C["road"],
                linewidth=2.3,
                alpha=0.95,
                solid_capstyle="round",
                zorder=3,
            )

            path = None
            if key in self.shortest_path_keys:
                path, = self.ax.plot(
                    [],
                    [],
                    color=C["cyan"],
                    linewidth=4.6,
                    alpha=0.92,
                    solid_capstyle="round",
                    zorder=4,
                )

            self.segment_artists[key] = (glow, road, path)

        self.intersection_scatter = self.ax.scatter(
            [],
            [],
            s=60,
            color=C["orange"],
            edgecolors=C["bg"],
            linewidths=1.2,
            zorder=8,
        )

        self.connection_scatter = self.ax.scatter(
            [],
            [],
            s=21,
            color=C["road"],
            edgecolors=C["bg"],
            linewidths=0.8,
            zorder=7,
        )

        self.vehicle_scatter = self.ax.scatter(
            [],
            [],
            s=54,
            color=C["white"],
            edgecolors=C["cyan"],
            linewidths=1.7,
            zorder=12,
        )

        self.build_position = 0.0
        self.build_finished = False
        self.make_vehicles()

        self.update_build_drawing()
        self.update_status_text()
        self.fig.canvas.draw_idle()

    def update_build_drawing(self):
        for segment in self.segments:
            key = (segment.start.id, segment.end.id)
            local_progress = self.build_position - segment.start.x
            local_progress = max(0.0, min(1.0, local_progress))

            xs, ys = segment.sample(30, local_progress)
            glow, road, path = self.segment_artists[key]

            if local_progress <= 0.0:
                glow.set_data([], [])
                road.set_data([], [])
                if path is not None:
                    path.set_data([], [])
                continue

            glow.set_data(xs, ys)
            road.set_data(xs, ys)

            if path is not None:
                path.set_data(xs, ys)

        self.update_visible_nodes()

    def update_visible_nodes(self):
        intersections_x = []
        intersections_y = []
        connections_x = []
        connections_y = []

        reveal_x = self.build_position + 0.02

        for node in self.nodes.values():
            if node.x > reveal_x:
                continue

            if node.type == NodeType.Intersection:
                intersections_x.append(node.x)
                intersections_y.append(node.y)
            elif node.type == NodeType.Connection:
                connections_x.append(node.x)
                connections_y.append(node.y)

        if intersections_x:
            self.intersection_scatter.set_offsets(
                list(zip(intersections_x, intersections_y))
            )
        else:
            self.intersection_scatter.set_offsets([])

        if connections_x:
            self.connection_scatter.set_offsets(
                list(zip(connections_x, connections_y))
            )
        else:
            self.connection_scatter.set_offsets([])

    # ------------------------------------------------------------------
    # VEHICLES
    # ------------------------------------------------------------------

    def make_vehicles(self):
        self.vehicles = []

        first_edges = self.forward_graph[self.start_node]

        if not first_edges:
            return

        for _ in range(self.vehicle_count):
            next_node, segment = self.rng.choice(first_edges)

            self.vehicles.append(
                {
                    "from": self.start_node,
                    "to": next_node,
                    "segment": segment,
                    "progress": self.rng.random() * 0.85,
                    "speed": self.rng.uniform(0.007, 0.014),
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
            self.vehicle_scatter.set_offsets([])
            return

        positions = []

        for vehicle in self.vehicles:
            vehicle["progress"] += vehicle["speed"] * self.traffic_speed

            if vehicle["progress"] >= 1.0:
                current = vehicle["to"]

                if current == self.end_node:
                    if not self.restart_vehicle(vehicle):
                        continue
                else:
                    choices = self.forward_graph[current]

                    if not choices:
                        if not self.restart_vehicle(vehicle):
                            continue
                    else:
                        next_node, segment = self.rng.choice(choices)
                        vehicle["from"] = current
                        vehicle["to"] = next_node
                        vehicle["segment"] = segment
                        vehicle["progress"] = 0.0

            positions.append(
                vehicle["segment"].point_at(vehicle["progress"])
            )

        self.vehicle_scatter.set_offsets(positions)

    # ------------------------------------------------------------------
    # STATE / CALLBACKS
    # ------------------------------------------------------------------

    def update_status_text(self):
        intersections = sum(
            node.type == NodeType.Intersection
            for node in self.nodes.values()
        )

        if self.build_finished:
            state = "TRAFFIC"
        elif self.paused:
            state = "PAUSED"
        else:
            percent = min(
                100,
                int(self.build_position / (self.cols - 1) * 100),
            )
            state = f"BUILD {percent:02d}%"

        self.header_status.set_text(
            f"seed {self.seed:05d}  •  "
            f"{self.route_count} routes  •  "
            f"{len(self.segments)} links"
        )

        self.stat_labels["state"].set_text(state)
        self.stat_labels["segments"].set_text(str(len(self.segments)))
        self.stat_labels["routes"].set_text(str(self.route_count))
        self.stat_labels["intersections"].set_text(str(intersections))
        self.stat_labels["vehicles"].set_text(str(self.vehicle_count))
        self.stat_labels["grid"].set_text(f"{self.cols}×{self.rows}")

    def update(self, _frame):
        if self.paused:
            return

        if not self.build_finished:
            # All routes grow together from left to right.
            self.build_position += 0.075 * self.build_speed

            if self.build_position >= self.cols - 1:
                self.build_position = self.cols - 1
                self.build_finished = True

            self.update_build_drawing()
            self.update_status_text()
        else:
            self.update_vehicles()

    def on_route_count(self, value):
        self.route_count = int(value)
        self.update_status_text()
        self.fig.canvas.draw_idle()

    def on_vehicle_count(self, value):
        self.vehicle_count = int(value)
        self.make_vehicles()
        self.update_status_text()
        self.fig.canvas.draw_idle()

    def on_speed_change(self, _value):
        self.build_speed = float(self.sliders["Construction"].val)
        self.traffic_speed = float(self.sliders["Traffic"].val)
        self.fig.canvas.draw_idle()

    def randomize(self, _event):
        self.route_count = int(self.sliders["Routes"].val)
        self.vehicle_count = int(self.sliders["Vehicles"].val)
        self.paused = False
        self.pause_button.label.set_text("PAUSE")

        self.generate_network(new_endpoints=True)
        self.rebuild_scene()
        self.fig.canvas.draw_idle()

    def new_roads_same_endpoints(self, _event):
        self.route_count = int(self.sliders["Routes"].val)
        self.vehicle_count = int(self.sliders["Vehicles"].val)
        self.paused = False
        self.pause_button.label.set_text("PAUSE")

        self.generate_network(new_endpoints=False)
        self.rebuild_scene()
        self.fig.canvas.draw_idle()

    def replay(self, _event):
        self.paused = False
        self.pause_button.label.set_text("PAUSE")
        self.build_position = 0.0
        self.build_finished = False
        self.make_vehicles()
        self.update_build_drawing()
        self.update_status_text()
        self.vehicle_scatter.set_offsets([])
        self.fig.canvas.draw_idle()

    def toggle_pause(self, _event):
        self.paused = not self.paused
        self.pause_button.label.set_text(
            "RESUME" if self.paused else "PAUSE"
        )
        self.update_status_text()
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


if __name__ == "__main__":
    RoadNetworkApp().show()
