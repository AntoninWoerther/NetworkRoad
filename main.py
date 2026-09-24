import heapq
import random
from collections import defaultdict
from functools import lru_cache

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
    "purple": "#B779FF",
    "white": "#F4F8FB",
    "muted": "#7890A3",
}


class RoadNetworkApp:
    def __init__(self):
        self.cols, self.rows = GRID_SIZE

        self.route_count = 5
        self.vehicle_count = 10
        self.build_speed = 1.0
        self.traffic_speed = 3.0

        self.pending_cols = self.cols
        self.pending_rows = self.rows
        self.pending_route_count = self.route_count

        self.paused = False
        self.seed = random.SystemRandom().randint(0, MAX_SEED)
        self.rng = random.Random(self.seed)

        self.nodes = {}
        self.routes = []
        self.segments = []
        self.segment_by_key = {}
        self.graph = defaultdict(list)
        self.node_usage = defaultdict(int)

        self.start_node = None
        self.end_node = None
        self.shortest_path_keys = set()
        self.shortest_distance = 0.0
        self.route_lengths = []

        self.built_segment_keys = set()
        self.build_route_index = 0
        self.build_edge_index = 0
        self.build_edge_progress = 0.0
        self.build_finished = False
        self.total_build_edges = 1
        self.completed_build_edges = 0

        self.vehicles = []
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

        ax = fig.add_axes([0.045, 0.245, 0.705, 0.605])
        ax.set_facecolor(C["bg"])
        ax.set_xlim(-0.9, self.cols - 0.1)
        ax.set_ylim(-0.9, self.rows - 0.1)
        ax.set_aspect("equal")
        ax.axis("off")

        return fig, ax

    def create_interface(self):
        self.fig.text(
            0.05, 0.955, "ROAD NETWORK",
            color=C["white"], fontsize=24, fontweight="bold",
        )
        self.fig.text(
            0.05, 0.922,
            "Sequential procedural routes  /  two-way traffic  /  deterministic seeds",
            color=C["muted"], fontsize=10,
        )
        self.header_status = self.fig.text(
            0.05, 0.882, "",
            color=C["cyan"], fontsize=9.5, family="monospace",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": C["panel"],
                "edgecolor": C["border"],
                "linewidth": 1,
            },
        )

        self.side_ax = self.fig.add_axes([0.775, 0.245, 0.195, 0.605])
        self.side_ax.set_facecolor(C["panel"])
        self.side_ax.set_xlim(0, 1)
        self.side_ax.set_ylim(0, 1)
        self.side_ax.set_xticks([])
        self.side_ax.set_yticks([])

        for spine in self.side_ax.spines.values():
            spine.set_color(C["border"])
            spine.set_linewidth(1.1)

        self.side_ax.text(
            0.09, 0.94, "NETWORK STATUS",
            color=C["white"], fontsize=12, fontweight="bold", va="top",
        )
        self.side_ax.text(
            0.09, 0.895, "LIVE SIMULATION",
            color=C["cyan"], fontsize=7.5, family="monospace", va="top",
        )
        self.side_ax.plot(
            [0.09, 0.91], [0.855, 0.855],
            color=C["border"], linewidth=1,
        )

        self.stat_labels = {}
        stat_rows = [
            ("state", "State"),
            ("seed", "Seed"),
            ("grid", "Grid"),
            ("segments", "Segments"),
            ("routes", "Routes"),
            ("intersections", "Junctions"),
            ("vehicles", "Vehicles"),
            ("shortest", "Shortest"),
            ("longest", "Longest"),
        ]

        y = 0.805
        for key, label in stat_rows:
            self.side_ax.text(
                0.09, y, label.upper(),
                color=C["muted"], fontsize=7.8,
                family="monospace", va="center",
            )
            self.stat_labels[key] = self.side_ax.text(
                0.91, y, "",
                color=C["white"], fontsize=8.9,
                family="monospace", ha="right", va="center",
            )
            y -= 0.044

        self.side_ax.plot(
            [0.09, 0.91], [0.385, 0.385],
            color=C["border"], linewidth=1,
        )
        self.side_ax.text(
            0.09, 0.345, "LEGEND",
            color=C["white"], fontsize=9.5, fontweight="bold",
        )

        legend = [
            (C["green"], "Start"),
            (C["red"], "End"),
            (C["orange"], "Junction"),
            (C["road"], "Road"),
            (C["cyan"], "Shortest path"),
            (C["cyan"], "Traffic START → END"),
            (C["purple"], "Traffic END → START"),
        ]

        y = 0.303
        for color, label in legend:
            self.side_ax.scatter([0.13], [y], s=52, color=color, zorder=2)
            self.side_ax.text(
                0.23, y, label,
                color=C["white"], fontsize=8.1, va="center",
            )
            y -= 0.039

        self.side_ax.text(
            0.09, 0.018,
            "ROUTES alternate START→END / END→START",
            color=C["cyan"], fontsize=6.9,
            family="monospace", va="bottom",
        )

        self.fig.text(
            0.05, 0.205, "CONFIGURATION",
            color=C["muted"], fontsize=8, family="monospace",
        )

        self.sliders = {}
        self.sliders["Routes"] = self.make_slider(
            [0.075, 0.155, 0.18, 0.019],
            "Routes", 1, 9, self.route_count, 1,
        )
        self.sliders["Vehicles"] = self.make_slider(
            [0.345, 0.155, 0.18, 0.019],
            "Vehicles", 2, 30, self.vehicle_count, 1,
        )
        self.sliders["Columns"] = self.make_slider(
            [0.075, 0.110, 0.18, 0.019],
            "Columns", 12, 36, self.cols, 1,
        )
        self.sliders["Rows"] = self.make_slider(
            [0.345, 0.110, 0.18, 0.019],
            "Rows", 8, 20, self.rows, 1,
        )
        self.sliders["Construction"] = self.make_slider(
            [0.075, 0.065, 0.18, 0.019],
            "Build", 0.25, 4.0, self.build_speed, None,
        )
        self.sliders["Traffic"] = self.make_slider(
            [0.345, 0.065, 0.18, 0.019],
            "Traffic", 0.25, 8.0, self.traffic_speed, None,
        )

        self.sliders["Routes"].on_changed(self.on_config_change)
        self.sliders["Columns"].on_changed(self.on_config_change)
        self.sliders["Rows"].on_changed(self.on_config_change)
        self.sliders["Vehicles"].on_changed(self.on_vehicle_count)
        self.sliders["Construction"].on_changed(self.on_speed_change)
        self.sliders["Traffic"].on_changed(self.on_speed_change)

        self.generate_button = self.make_button(
            [0.58, 0.152, 0.105, 0.044], "GENERATE"
        )
        self.random_button = self.make_button(
            [0.695, 0.152, 0.095, 0.044], "RANDOMIZE"
        )
        self.pause_button = self.make_button(
            [0.80, 0.152, 0.075, 0.044], "PAUSE"
        )
        self.replay_button = self.make_button(
            [0.885, 0.152, 0.075, 0.044], "REPLAY"
        )
        self.new_path_button = self.make_button(
            [0.58, 0.098, 0.19, 0.038],
            "NEW ROUTES · SAME ENDS",
        )

        seed_ax = self.fig.add_axes([0.79, 0.098, 0.105, 0.038])
        seed_ax.set_facecolor(C["panel"])
        self.seed_box = TextBox(
            seed_ax, "", initial=str(self.seed),
            color=C["panel"], hovercolor=C["panel_2"],
        )
        self.seed_box.text_disp.set_color(C["white"])
        self.seed_box.text_disp.set_fontfamily("monospace")
        self.seed_box.text_disp.set_fontsize(9)
        for spine in seed_ax.spines.values():
            spine.set_color(C["border"])

        self.load_seed_button = self.make_button(
            [0.905, 0.098, 0.055, 0.038], "LOAD"
        )
        self.fig.text(
            0.79, 0.139, "SEED",
            color=C["muted"], fontsize=7.3, family="monospace",
        )
        self.seed_feedback = self.fig.text(
            0.58, 0.058,
            "Change grid/routes, then press GENERATE",
            color=C["muted"], fontsize=7.2,
        )

        self.generate_button.on_clicked(self.generate_from_controls)
        self.random_button.on_clicked(self.randomize)
        self.pause_button.on_clicked(self.toggle_pause)
        self.replay_button.on_clicked(self.replay)
        self.new_path_button.on_clicked(self.new_routes_same_endpoints)
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

    @staticmethod
    def segment_key(a, b):
        return tuple(sorted((a.id, b.id)))

    def generate_network(self, seed=None, new_endpoints=True):
        old_start_y = self.start_node.y if self.start_node is not None else None
        old_end_y = self.end_node.y if self.end_node is not None else None

        if seed is not None:
            self.seed = int(seed) % (MAX_SEED + 1)

        self.rng = random.Random(self.seed)
        self.make_grid()

        self.routes = []
        self.segments = []
        self.segment_by_key = {}
        self.graph = defaultdict(list)
        self.node_usage = defaultdict(int)
        self.route_lengths = []
        self.shortest_path_keys = set()
        self.shortest_distance = 0.0

        if (
            new_endpoints
            or old_start_y is None
            or old_end_y is None
            or not 0 <= old_start_y < self.rows
            or not 0 <= old_end_y < self.rows
        ):
            start_y = self.rng.randint(2, self.rows - 3)
            end_y = self.rng.randint(2, self.rows - 3)
        else:
            start_y = old_start_y
            end_y = old_end_y

        self.start_node = self.nodes[(0, start_y)]
        self.end_node = self.nodes[(self.cols - 1, end_y)]

        for route_index in range(self.route_count):
            if route_index % 2 == 0:
                source = self.start_node
                target = self.end_node
            else:
                source = self.end_node
                target = self.start_node

            route = self.generate_route(route_index, source, target)
            self.routes.append(route)

            for node in route:
                self.node_usage[node] += 1

            route_length = 0.0
            for a, b in zip(route, route[1:]):
                segment = self.register_segment(a, b)
                route_length += segment.length

            self.route_lengths.append(route_length)

        self.total_build_edges = max(
            1,
            sum(max(0, len(route) - 1) for route in self.routes),
        )

        self.classify_nodes()
        self.compute_shortest_path()

    def generate_route(self, route_index, source, target):
        """
        Generate a route from source to target. Horizontal progress always moves
        toward the target, while rare vertical detours are allowed. No turn has
        an interior angle below 90°, and no direction repeats more than twice.
        """
        direction_sign = 1 if target.x > source.x else -1
        target_x = target.x
        target_y = target.y

        delta_y = {
            "U": 1,
            "D": -1,
            "F": 0,
            "FU": 1,
            "FD": -1,
        }
        vectors = {
            "U": (0, 1),
            "D": (0, -1),
            "F": (direction_sign, 0),
            "FU": (direction_sign, 1),
            "FD": (direction_sign, -1),
        }

        def angle_is_valid(previous_direction, next_direction):
            if previous_direction == "START":
                return True
            ax, ay = vectors[previous_direction]
            bx, by = vectors[next_direction]
            return ax * bx + ay * by >= 0

        if self.route_count <= 1:
            detour_profile = 0.0
        else:
            detour_profile = route_index / (self.route_count - 1)

        vertical_chance = 0.008 + 0.035 * detour_profile
        vertical_patterns = [
            ((), 1.0 - vertical_chance),
            (("U",), vertical_chance * 0.485),
            (("D",), vertical_chance * 0.485),
            (("U", "U"), vertical_chance * 0.015),
            (("D", "D"), vertical_chance * 0.015),
        ]
        forward_moves = ("F", "FU", "FD")

        def next_repeat(last_direction, repeat_count, direction):
            count = repeat_count + 1 if direction == last_direction else 1
            if count > 2:
                return None
            return count

        def simulate_action(x, y, last_direction, repeat_count, verticals, forward):
            local_y = y
            local_last = last_direction
            local_repeat = repeat_count
            nodes = []
            directions = []

            for direction in verticals:
                if not angle_is_valid(local_last, direction):
                    return None

                repeat = next_repeat(local_last, local_repeat, direction)
                if repeat is None:
                    return None

                local_y += delta_y[direction]
                if not 0 <= local_y < self.rows:
                    return None

                nodes.append(self.nodes[(x, local_y)])
                directions.append(direction)
                local_last = direction
                local_repeat = repeat

            if not angle_is_valid(local_last, forward):
                return None

            repeat = next_repeat(local_last, local_repeat, forward)
            if repeat is None:
                return None

            next_y = local_y + delta_y[forward]
            if not 0 <= next_y < self.rows:
                return None

            next_x = x + direction_sign
            nodes.append(self.nodes[(next_x, next_y)])
            directions.append(forward)

            return nodes, directions, next_x, next_y, forward, repeat

        @lru_cache(maxsize=None)
        def can_finish(x, y, last_direction, repeat_count):
            if x == target_x:
                return y == target_y

            for verticals, _pattern_weight in vertical_patterns:
                for forward in forward_moves:
                    result = simulate_action(
                        x, y, last_direction, repeat_count,
                        verticals, forward,
                    )
                    if result is None:
                        continue

                    _nodes, _directions, nx, ny, nl, nc = result
                    if nx == target_x and ny != target_y:
                        continue
                    if can_finish(nx, ny, nl, nc):
                        return True

            return False

        route = [source]
        current = source
        last_direction = "START"
        repeat_count = 0

        while current.x != target_x:
            candidates = []
            weights = []

            for verticals, pattern_weight in vertical_patterns:
                for forward in forward_moves:
                    result = simulate_action(
                        current.x,
                        current.y,
                        last_direction,
                        repeat_count,
                        verticals,
                        forward,
                    )
                    if result is None:
                        continue

                    nodes, directions, nx, ny, nl, nc = result
                    if nx == target_x and ny != target_y:
                        continue
                    if not can_finish(nx, ny, nl, nc):
                        continue

                    forward_from = current if len(nodes) == 1 else nodes[-2]
                    forward_to = nodes[-1]
                    crossing = self.edge_would_cross(
                        forward_from,
                        forward_to,
                    )

                    before_distance = abs(target_y - current.y)
                    after_distance = abs(target_y - ny)
                    improvement = before_distance - after_distance

                    if forward == "F":
                        forward_weight = 1.50
                    elif improvement > 0:
                        forward_weight = 1.75
                    elif improvement == 0:
                        forward_weight = 1.00
                    else:
                        forward_weight = 0.35 + 0.55 * detour_profile

                    reuse = sum(
                        self.node_usage.get(node, 0)
                        for node in nodes
                    )
                    weight = pattern_weight * forward_weight
                    weight /= 1.0 + reuse * 0.50

                    if crossing:
                        weight *= 0.08
                    if route_index == 0 and verticals:
                        weight *= 0.10

                    candidates.append((nodes, directions, nl, nc))
                    weights.append(max(weight, 1e-9))

            if not candidates:
                raise RuntimeError(
                    "No valid route can reach the requested endpoint."
                )

            selected = self.rng.choices(
                range(len(candidates)),
                weights=weights,
                k=1,
            )[0]

            (
                chosen_nodes,
                _chosen_directions,
                last_direction,
                repeat_count,
            ) = candidates[selected]

            route.extend(chosen_nodes)
            current = chosen_nodes[-1]

        if current != target:
            raise RuntimeError(
                "Generated route did not finish on its target endpoint."
            )

        for previous, center, following in zip(
            route,
            route[1:],
            route[2:],
        ):
            incoming = (
                center.x - previous.x,
                center.y - previous.y,
            )
            outgoing = (
                following.x - center.x,
                following.y - center.y,
            )
            if incoming[0] * outgoing[0] + incoming[1] * outgoing[1] < 0:
                raise RuntimeError(
                    "Generated route contains an angle below 90 degrees."
                )

        return route

    def edge_would_cross(self, a, b):
        """Reject X-shaped diagonal crossings between the same two columns."""
        if a.x == b.x:
            return False

        low_x = min(a.x, b.x)
        high_x = max(a.x, b.x)
        a_low = a if a.x == low_x else b
        a_high = b if b.x == high_x else a

        for segment in self.segments:
            if segment.start.x == segment.end.x:
                continue

            seg_low_x = min(segment.start.x, segment.end.x)
            seg_high_x = max(segment.start.x, segment.end.x)
            if seg_low_x != low_x or seg_high_x != high_x:
                continue

            s_low = (
                segment.start
                if segment.start.x == low_x
                else segment.end
            )
            s_high = (
                segment.end
                if segment.end.x == high_x
                else segment.start
            )

            d0 = a_low.y - s_low.y
            d1 = a_high.y - s_high.y
            if d0 * d1 < 0:
                return True

        return False

    def register_segment(self, a, b):
        key = self.segment_key(a, b)
        if key in self.segment_by_key:
            return self.segment_by_key[key]

        segment = Segment(a, b)
        self.segment_by_key[key] = segment
        self.segments.append(segment)
        self.graph[a].append((b, segment))
        self.graph[b].append((a, segment))
        return segment

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

            for neighbour, segment in self.graph[node]:
                candidate = distance + segment.length

                if candidate < distances.get(
                    neighbour,
                    float("inf"),
                ):
                    distances[neighbour] = candidate
                    previous[neighbour] = (node, segment)
                    heapq.heappush(
                        queue,
                        (candidate, neighbour.id, neighbour),
                    )

        self.shortest_distance = distances.get(
            self.end_node,
            float("inf"),
        )

        node = self.end_node
        while node in previous:
            parent, segment = previous[node]
            self.shortest_path_keys.add(
                self.segment_key(segment.start, segment.end)
            )
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
        self.ax.scatter(
            xs, ys,
            s=9,
            color=C["grid_bright"],
            alpha=0.52,
            zorder=1,
        )

        self.road_glow = LineCollection(
            [],
            colors=C["cyan"],
            linewidths=8.0,
            alpha=0.07,
            zorder=2,
        )
        self.road_lines = LineCollection(
            [],
            colors=C["road"],
            linewidths=2.15,
            alpha=0.94,
            zorder=3,
        )
        self.path_glow = LineCollection(
            [],
            colors=C["cyan"],
            linewidths=4.4,
            alpha=0.90,
            zorder=4,
        )
        self.ax.add_collection(self.road_glow)
        self.ax.add_collection(self.road_lines)
        self.ax.add_collection(self.path_glow)

        self.start_marker = self.ax.scatter(
            [self.start_node.x],
            [self.start_node.y],
            s=105,
            color=C["green"],
            edgecolors=C["white"],
            linewidths=1.0,
            zorder=10,
        )
        self.end_marker = self.ax.scatter(
            [self.end_node.x],
            [self.end_node.y],
            s=105,
            color=C["red"],
            edgecolors=C["white"],
            linewidths=1.0,
            zorder=10,
        )
        self.start_ring = self.ax.scatter(
            [self.start_node.x],
            [self.start_node.y],
            s=220,
            facecolors="none",
            edgecolors=C["green"],
            linewidths=1.1,
            alpha=0.25,
            zorder=9,
        )
        self.end_ring = self.ax.scatter(
            [self.end_node.x],
            [self.end_node.y],
            s=220,
            facecolors="none",
            edgecolors=C["red"],
            linewidths=1.1,
            alpha=0.25,
            zorder=9,
        )

        self.ax.text(
            self.start_node.x + 0.25,
            self.start_node.y + 0.35,
            "START",
            color=C["green"],
            fontsize=7.5,
            fontweight="bold",
            zorder=11,
        )
        self.ax.text(
            self.end_node.x - 0.25,
            self.end_node.y + 0.35,
            "END",
            color=C["red"],
            fontsize=7.5,
            fontweight="bold",
            ha="right",
            zorder=11,
        )

        self.intersection_scatter = self.ax.scatter(
            [], [],
            s=58,
            color=C["orange"],
            edgecolors=C["bg"],
            linewidths=1.2,
            zorder=8,
        )
        self.connection_scatter = self.ax.scatter(
            [], [],
            s=17,
            color=C["road"],
            edgecolors=C["bg"],
            linewidths=0.7,
            zorder=7,
        )
        self.builder_glow = self.ax.scatter(
            [], [],
            s=170,
            color=C["cyan"],
            alpha=0.12,
            edgecolors="none",
            zorder=11,
        )
        self.builder_scatter = self.ax.scatter(
            [], [],
            s=46,
            color=C["cyan"],
            edgecolors=C["white"],
            linewidths=0.8,
            zorder=12,
        )

        self.vehicle_out_glow = self.ax.scatter(
            [], [],
            s=145,
            color=C["cyan"],
            alpha=0.12,
            edgecolors="none",
            zorder=11,
        )
        self.vehicle_out = self.ax.scatter(
            [], [],
            s=40,
            color=C["cyan"],
            edgecolors=C["white"],
            linewidths=0.7,
            zorder=12,
        )
        self.vehicle_back_glow = self.ax.scatter(
            [], [],
            s=145,
            color=C["purple"],
            alpha=0.12,
            edgecolors="none",
            zorder=11,
        )
        self.vehicle_back = self.ax.scatter(
            [], [],
            s=40,
            color=C["purple"],
            edgecolors=C["white"],
            linewidths=0.7,
            zorder=12,
        )

        self.reset_build_state()
        self.make_vehicles()
        self.update_builder_color()
        self.update_build_drawing()
        self.update_status_text()
        self.fig.canvas.draw_idle()

    def reset_build_state(self):
        self.built_segment_keys = set()
        self.build_route_index = 0
        self.build_edge_index = 0
        self.build_edge_progress = 0.0
        self.build_finished = len(self.routes) == 0
        self.completed_build_edges = 0

    @staticmethod
    def interpolate_nodes(a, b, t):
        t = max(0.0, min(1.0, float(t)))
        return (
            a.x + (b.x - a.x) * t,
            a.y + (b.y - a.y) * t,
        )

    def current_build_edge(self):
        if (
            self.build_finished
            or self.build_route_index >= len(self.routes)
        ):
            return None

        route = self.routes[self.build_route_index]
        if self.build_edge_index >= len(route) - 1:
            return None

        return (
            route[self.build_edge_index],
            route[self.build_edge_index + 1],
        )

    def advance_build(self):
        edge = self.current_build_edge()
        if edge is None:
            self.finish_or_advance_route()
            return

        a, b = edge
        distance = max(
            np.hypot(b.x - a.x, b.y - a.y),
            1e-9,
        )
        self.build_edge_progress += (
            0.085 * self.build_speed
        ) / distance

        if self.build_edge_progress >= 1.0:
            self.built_segment_keys.add(
                self.segment_key(a, b)
            )
            self.completed_build_edges += 1
            self.build_edge_progress = 0.0
            self.build_edge_index += 1

            route = self.routes[self.build_route_index]
            if self.build_edge_index >= len(route) - 1:
                self.finish_or_advance_route()

    def finish_or_advance_route(self):
        if self.build_route_index + 1 >= len(self.routes):
            self.build_finished = True
            self.builder_scatter.set_offsets(
                np.empty((0, 2))
            )
            self.builder_glow.set_offsets(
                np.empty((0, 2))
            )
            return

        self.build_route_index += 1
        self.build_edge_index = 0
        self.build_edge_progress = 0.0
        self.update_builder_color()

    def update_builder_color(self):
        if self.build_route_index >= len(self.routes):
            return

        route = self.routes[self.build_route_index]
        color = (
            C["cyan"]
            if route[0] == self.start_node
            else C["purple"]
        )
        self.builder_scatter.set_facecolor(color)
        self.builder_glow.set_facecolor(color)

    def update_build_drawing(self):
        full_roads = []
        shortest = []

        for segment in self.segments:
            if segment.key not in self.built_segment_keys:
                continue

            line = [
                (segment.start.x, segment.start.y),
                (segment.end.x, segment.end.y),
            ]
            full_roads.append(line)

            if segment.key in self.shortest_path_keys:
                shortest.append(line)

        active_edge = self.current_build_edge()
        if active_edge is not None:
            a, b = active_edge
            current = self.interpolate_nodes(
                a,
                b,
                self.build_edge_progress,
            )
            active_key = self.segment_key(a, b)

            if active_key not in self.built_segment_keys:
                full_roads.append([
                    (a.x, a.y),
                    current,
                ])
                if active_key in self.shortest_path_keys:
                    shortest.append([
                        (a.x, a.y),
                        current,
                    ])

            offsets = np.array([current])
            self.builder_scatter.set_offsets(offsets)
            self.builder_glow.set_offsets(offsets)
        else:
            self.builder_scatter.set_offsets(
                np.empty((0, 2))
            )
            self.builder_glow.set_offsets(
                np.empty((0, 2))
            )

        self.road_glow.set_segments(full_roads)
        self.road_lines.set_segments(full_roads)
        self.path_glow.set_segments(shortest)
        self.update_visible_nodes()

    def update_visible_nodes(self):
        visible_nodes = {
            self.start_node,
            self.end_node,
        }

        for segment in self.segments:
            if segment.key in self.built_segment_keys:
                visible_nodes.add(segment.start)
                visible_nodes.add(segment.end)

        active = self.current_build_edge()
        if active is not None:
            visible_nodes.add(active[0])

        junctions = []
        connections = []

        for node in visible_nodes:
            if node.type == NodeType.Intersection:
                junctions.append((node.x, node.y))
            elif node.type == NodeType.Connection:
                connections.append((node.x, node.y))

        self.intersection_scatter.set_offsets(
            junctions
            if junctions
            else np.empty((0, 2))
        )
        self.connection_scatter.set_offsets(
            connections
            if connections
            else np.empty((0, 2))
        )

    # ------------------------------------------------------------------
    # VEHICLES
    # ------------------------------------------------------------------

    def route_for_traffic_direction(self, route_index, direction):
        route = self.routes[route_index]
        desired_start = (
            self.start_node
            if direction == 1
            else self.end_node
        )

        if route[0] == desired_start:
            return route

        return list(reversed(route))

    def make_vehicles(self):
        self.vehicles = []

        if not self.routes:
            return

        for index in range(self.vehicle_count):
            direction = 1 if index % 2 == 0 else -1
            route_index = index % len(self.routes)
            path = self.route_for_traffic_direction(
                route_index,
                direction,
            )

            if len(path) < 2:
                continue

            self.vehicles.append({
                "direction": direction,
                "route_index": route_index,
                "path": path,
                "edge_index": 0,
                "progress": (
                    index
                    / max(self.vehicle_count, 1)
                ) * 0.85,
                "speed": self.rng.uniform(
                    0.040,
                    0.070,
                ),
            })

    def restart_vehicle(self, vehicle):
        vehicle["route_index"] = self.rng.randrange(
            len(self.routes)
        )
        vehicle["path"] = self.route_for_traffic_direction(
            vehicle["route_index"],
            vehicle["direction"],
        )
        vehicle["edge_index"] = 0
        vehicle["progress"] = 0.0

    def update_vehicles(self):
        if not self.build_finished:
            empty = np.empty((0, 2))
            self.vehicle_out.set_offsets(empty)
            self.vehicle_out_glow.set_offsets(empty)
            self.vehicle_back.set_offsets(empty)
            self.vehicle_back_glow.set_offsets(empty)
            return

        outbound = []
        inbound = []

        for vehicle in self.vehicles:
            remaining = (
                vehicle["speed"]
                * self.traffic_speed
            )

            while remaining > 0:
                path = vehicle["path"]
                edge_index = vehicle["edge_index"]

                if edge_index >= len(path) - 1:
                    self.restart_vehicle(vehicle)
                    continue

                a = path[edge_index]
                b = path[edge_index + 1]
                length = max(
                    np.hypot(
                        b.x - a.x,
                        b.y - a.y,
                    ),
                    1e-9,
                )

                edge_remaining = (
                    1.0 - vehicle["progress"]
                ) * length

                if remaining < edge_remaining:
                    vehicle["progress"] += (
                        remaining / length
                    )
                    remaining = 0.0
                else:
                    remaining -= edge_remaining
                    vehicle["edge_index"] += 1
                    vehicle["progress"] = 0.0

                    if (
                        vehicle["edge_index"]
                        >= len(path) - 1
                    ):
                        self.restart_vehicle(vehicle)

            path = vehicle["path"]
            edge_index = min(
                vehicle["edge_index"],
                len(path) - 2,
            )
            point = self.interpolate_nodes(
                path[edge_index],
                path[edge_index + 1],
                vehicle["progress"],
            )

            if vehicle["direction"] == 1:
                outbound.append(point)
            else:
                inbound.append(point)

        out_offsets = (
            outbound
            if outbound
            else np.empty((0, 2))
        )
        back_offsets = (
            inbound
            if inbound
            else np.empty((0, 2))
        )

        self.vehicle_out.set_offsets(out_offsets)
        self.vehicle_out_glow.set_offsets(out_offsets)
        self.vehicle_back.set_offsets(back_offsets)
        self.vehicle_back_glow.set_offsets(back_offsets)

    # ------------------------------------------------------------------
    # STATE / CALLBACKS
    # ------------------------------------------------------------------

    def update_status_text(self):
        junctions = sum(
            node.type == NodeType.Intersection
            for node in self.nodes.values()
        )

        if self.paused:
            state = "PAUSED"
        elif self.build_finished:
            state = "TRAFFIC"
        else:
            percent = min(
                100,
                int(
                    100
                    * self.completed_build_edges
                    / max(self.total_build_edges, 1)
                ),
            )
            direction = (
                "→"
                if self.build_route_index % 2 == 0
                else "←"
            )
            state = (
                f"BUILD {self.build_route_index + 1}/"
                f"{self.route_count} {direction} {percent:02d}%"
            )

        self.header_status.set_text(
            f"SEED {self.seed:09d}   •   "
            f"{self.route_count} ROUTES   •   "
            f"GRID {self.cols}×{self.rows}"
        )

        self.stat_labels["state"].set_text(state)
        self.stat_labels["seed"].set_text(
            f"{self.seed:09d}"
        )
        self.stat_labels["grid"].set_text(
            f"{self.cols}×{self.rows}"
        )
        self.stat_labels["segments"].set_text(
            str(len(self.segments))
        )
        self.stat_labels["routes"].set_text(
            str(self.route_count)
        )
        self.stat_labels["intersections"].set_text(
            str(junctions)
        )
        self.stat_labels["vehicles"].set_text(
            str(self.vehicle_count)
        )
        self.stat_labels["shortest"].set_text(
            f"{self.shortest_distance:.1f}"
            if np.isfinite(self.shortest_distance)
            else "--"
        )
        self.stat_labels["longest"].set_text(
            f"{max(self.route_lengths):.1f}"
            if self.route_lengths
            else "--"
        )

    def update(self, _frame):
        self.frame_counter += 1

        pulse = (
            220
            + 18 * np.sin(
                self.frame_counter * 0.08
            )
        )
        self.start_ring.set_sizes([pulse])
        self.end_ring.set_sizes([pulse])

        if self.paused:
            return

        if not self.build_finished:
            self.advance_build()
            self.update_build_drawing()

            if (
                self.frame_counter % 3 == 0
                or self.build_finished
            ):
                self.update_status_text()
        else:
            self.update_vehicles()

    def on_config_change(self, _value):
        self.pending_route_count = int(
            self.sliders["Routes"].val
        )
        self.pending_cols = int(
            self.sliders["Columns"].val
        )
        self.pending_rows = int(
            self.sliders["Rows"].val
        )

        self.seed_feedback.set_text(
            "Configuration changed · press GENERATE"
        )
        self.seed_feedback.set_color(
            C["orange"]
        )

    def on_vehicle_count(self, value):
        self.vehicle_count = int(value)
        self.make_vehicles()
        self.update_status_text()
        self.fig.canvas.draw_idle()

    def on_speed_change(self, _value):
        self.build_speed = float(
            self.sliders["Construction"].val
        )
        self.traffic_speed = float(
            self.sliders["Traffic"].val
        )

    def apply_pending_configuration(self):
        self.route_count = self.pending_route_count
        self.cols = self.pending_cols
        self.rows = self.pending_rows

    def generate_from_controls(self, _event=None):
        self.apply_pending_configuration()
        self.paused = False
        self.pause_button.label.set_text(
            "PAUSE"
        )

        self.generate_network(
            seed=self.seed,
            new_endpoints=True,
        )
        self.rebuild_scene()

        self.seed_feedback.set_text(
            "Generated with current seed"
        )
        self.seed_feedback.set_color(
            C["green"]
        )

    def randomize(self, _event=None):
        self.apply_pending_configuration()
        self.seed = (
            random.SystemRandom()
            .randint(0, MAX_SEED)
        )
        self.paused = False
        self.pause_button.label.set_text(
            "PAUSE"
        )

        self.generate_network(
            seed=self.seed,
            new_endpoints=True,
        )
        self.rebuild_scene()
        self.sync_seed_box()

        self.seed_feedback.set_text(
            "New random seed generated"
        )
        self.seed_feedback.set_color(
            C["green"]
        )

    def new_routes_same_endpoints(self, _event=None):
        dimensions_changed = (
            self.pending_cols != self.cols
            or self.pending_rows != self.rows
        )

        self.pending_route_count = int(
            self.sliders["Routes"].val
        )
        self.route_count = (
            self.pending_route_count
        )
        self.seed = (
            random.SystemRandom()
            .randint(0, MAX_SEED)
        )
        self.paused = False
        self.pause_button.label.set_text(
            "PAUSE"
        )

        if dimensions_changed:
            self.apply_pending_configuration()
            new_endpoints = True
            message = (
                "Grid changed · new endpoints generated"
            )
        else:
            new_endpoints = False
            message = (
                "New alternating routes · "
                "endpoints preserved"
            )

        self.generate_network(
            seed=self.seed,
            new_endpoints=new_endpoints,
        )
        self.rebuild_scene()
        self.sync_seed_box()

        self.seed_feedback.set_text(message)
        self.seed_feedback.set_color(
            C["cyan"]
        )

    def replay(self, _event=None):
        self.paused = False
        self.pause_button.label.set_text(
            "PAUSE"
        )

        self.reset_build_state()
        self.make_vehicles()
        self.update_builder_color()
        self.update_build_drawing()
        self.update_status_text()
        self.fig.canvas.draw_idle()

    def toggle_pause(self, _event=None):
        self.paused = not self.paused
        self.pause_button.label.set_text(
            "RESUME"
            if self.paused
            else "PAUSE"
        )
        self.update_status_text()
        self.fig.canvas.draw_idle()

    def load_seed(self, _event=None):
        raw = self.seed_box.text.strip()

        try:
            seed = int(raw)
            if seed < 0:
                raise ValueError
        except ValueError:
            self.seed_feedback.set_text(
                "Invalid seed · use a positive integer"
            )
            self.seed_feedback.set_color(
                C["red"]
            )
            return

        self.apply_pending_configuration()
        self.seed = seed % (
            MAX_SEED + 1
        )
        self.paused = False
        self.pause_button.label.set_text(
            "PAUSE"
        )

        self.generate_network(
            seed=self.seed,
            new_endpoints=True,
        )
        self.rebuild_scene()
        self.sync_seed_box()

        self.seed_feedback.set_text(
            "Seed loaded · network reproduced"
        )
        self.seed_feedback.set_color(
            C["green"]
        )

    def sync_seed_box(self):
        if not hasattr(
            self,
            "seed_box",
        ):
            return

        previous = (
            self.seed_box.eventson
        )
        self.seed_box.eventson = False
        self.seed_box.set_val(
            str(self.seed)
        )
        self.seed_box.eventson = previous

    def show(self):
        plt.show()


if __name__ == "__main__":
    RoadNetworkApp().show()
