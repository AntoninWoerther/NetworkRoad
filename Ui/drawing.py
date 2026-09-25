import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

from node import NodeType
from config import C


class DrawingManager:
    """Gère tout ce qui est affiché sur le canvas matplotlib."""

    def __init__(self, app):
        self.app = app

    # ── Scène principale ──────────────────────────────────────────────

    def rebuild_scene(self):
        app = self.app
        app.ax.clear()
        app.sign_patches = []
        app.ax.set_facecolor(C["bg"])
        app.ax.set_xlim(-0.9, app.cols - 0.1)
        app.ax.set_ylim(-0.9, app.rows - 0.1)
        app.ax.set_aspect("equal")
        app.ax.axis("off")

        for y in range(app.rows):
            app.ax.plot([-0.2, app.cols - 0.8], [y, y],
                        color=C["grid"], linewidth=0.35, alpha=0.18, zorder=0)

        xs = [n.x for n in app.nodes.values()]
        ys = [n.y for n in app.nodes.values()]
        app.ax.scatter(xs, ys, s=9, color=C["grid_bright"], alpha=0.52, zorder=1)

        app.road_glow  = LineCollection([], colors=C["cyan"], linewidths=8.0,  alpha=0.07, zorder=2)
        app.road_lines = LineCollection([], colors=C["road"], linewidths=2.15, alpha=0.94, zorder=3)
        app.path_glow  = LineCollection([], colors=C["cyan"], linewidths=4.4,  alpha=0.90, zorder=4)
        for col in (app.road_glow, app.road_lines, app.path_glow):
            app.ax.add_collection(col)

        app.start_marker = app.ax.scatter(
            [app.start_node.x], [app.start_node.y],
            s=105, color=C["green"], edgecolors=C["white"], linewidths=1.0, zorder=10)
        app.end_marker = app.ax.scatter(
            [app.end_node.x], [app.end_node.y],
            s=105, color=C["red"], edgecolors=C["white"], linewidths=1.0, zorder=10)
        app.start_ring = app.ax.scatter(
            [app.start_node.x], [app.start_node.y],
            s=220, facecolors="none", edgecolors=C["green"], linewidths=1.1, alpha=0.25, zorder=9)
        app.end_ring = app.ax.scatter(
            [app.end_node.x], [app.end_node.y],
            s=220, facecolors="none", edgecolors=C["red"], linewidths=1.1, alpha=0.25, zorder=9)

        app.ax.text(app.start_node.x + 0.25, app.start_node.y + 0.35, "START",
                    color=C["green"], fontsize=7.5, fontweight="bold", zorder=11)
        app.ax.text(app.end_node.x - 0.25, app.end_node.y + 0.35, "END",
                    color=C["red"], fontsize=7.5, fontweight="bold", ha="right", zorder=11)

        app.intersection_scatter = app.ax.scatter(
            [], [], s=58, color=C["orange"], edgecolors=C["bg"], linewidths=1.2, zorder=8)
        app.connection_scatter = app.ax.scatter(
            [], [], s=17, color=C["road"], edgecolors=C["bg"], linewidths=0.7, zorder=7)
        app.builder_glow = app.ax.scatter(
            [], [], s=170, color=C["cyan"], alpha=0.12, edgecolors="none", zorder=11)
        app.builder_scatter = app.ax.scatter(
            [], [], s=46, color=C["cyan"], edgecolors=C["white"], linewidths=0.8, zorder=12)

        app.vehicle_out_glow = app.ax.scatter(
            [], [], s=145, color=C["cyan"], alpha=0.12, edgecolors="none", zorder=11)
        app.vehicle_out = app.ax.scatter(
            [], [], s=40, color=C["cyan"], edgecolors=C["white"], linewidths=0.7, zorder=12)
        app.vehicle_back_glow = app.ax.scatter(
            [], [], s=145, color=C["purple"], alpha=0.12, edgecolors="none", zorder=11)
        app.vehicle_back = app.ax.scatter(
            [], [], s=40, color=C["purple"], edgecolors=C["white"], linewidths=0.7, zorder=12)

    # ── Construction animée ───────────────────────────────────────────

    def reset_build_state(self):
        app = self.app
        app.built_segment_keys    = set()
        app.build_route_index     = 0
        app.build_edge_index      = 0
        app.build_edge_progress   = 0.0
        app.build_finished        = len(app.routes) == 0
        app.completed_build_edges = 0

    def current_build_edge(self):
        app = self.app
        if app.build_finished or app.build_route_index >= len(app.routes):
            return None
        route = app.routes[app.build_route_index]
        if app.build_edge_index >= len(route) - 1:
            return None
        return route[app.build_edge_index], route[app.build_edge_index + 1]

    def advance_build(self):
        app  = self.app
        edge = self.current_build_edge()
        if edge is None:
            self.finish_or_advance_route()
            return

        a, b = edge
        dist  = max(np.hypot(b.x - a.x, b.y - a.y), 1e-9)
        app.build_edge_progress += (0.085 * app.build_speed) / dist

        if app.build_edge_progress >= 1.0:
            app.built_segment_keys.add(app.network.segment_key(a, b))
            app.completed_build_edges += 1
            app.build_edge_progress    = 0.0
            app.build_edge_index      += 1

            route = app.routes[app.build_route_index]
            if app.build_edge_index >= len(route) - 1:
                self.finish_or_advance_route()

    def finish_or_advance_route(self):
        app = self.app
        if app.build_route_index + 1 >= len(app.routes):
            app.build_finished = True
            app.builder_scatter.set_offsets(np.empty((0, 2)))
            app.builder_glow.set_offsets(np.empty((0, 2)))
            self.draw_speed_signs()
            return
        app.build_route_index  += 1
        app.build_edge_index    = 0
        app.build_edge_progress = 0.0
        self.update_builder_color()

    def update_builder_color(self):
        app = self.app
        if app.build_route_index >= len(app.routes):
            return
        color = C["cyan"] if app.routes[app.build_route_index][0] == app.start_node else C["purple"]
        app.builder_scatter.set_facecolor(color)
        app.builder_glow.set_facecolor(color)

    def update_build_drawing(self):
        app        = self.app
        full_roads = []
        shortest   = []

        for seg in app.segments:
            if seg.key not in app.built_segment_keys:
                continue
            line = [(seg.start.x, seg.start.y), (seg.end.x, seg.end.y)]
            full_roads.append(line)
            if seg.key in app.shortest_path_keys:
                shortest.append(line)

        edge = self.current_build_edge()
        if edge is not None:
            a, b       = edge
            cur        = self._interpolate(a, b, app.build_edge_progress)
            active_key = app.network.segment_key(a, b)

            if active_key not in app.built_segment_keys:
                line = [(a.x, a.y), cur]
                full_roads.append(line)
                if active_key in app.shortest_path_keys:
                    shortest.append(line)

            app.builder_scatter.set_offsets(np.array([cur]))
            app.builder_glow.set_offsets(np.array([cur]))
        else:
            app.builder_scatter.set_offsets(np.empty((0, 2)))
            app.builder_glow.set_offsets(np.empty((0, 2)))

        app.road_glow.set_segments(full_roads)
        app.road_lines.set_segments(full_roads)
        app.path_glow.set_segments(shortest)
        self.update_visible_nodes()

    def update_visible_nodes(self):
        app     = self.app
        visible = {app.start_node, app.end_node}

        for seg in app.segments:
            if seg.key in app.built_segment_keys:
                visible.add(seg.start)
                visible.add(seg.end)

        edge = self.current_build_edge()
        if edge is not None:
            visible.add(edge[0])

        junctions   = [(n.x, n.y) for n in visible if n.type == NodeType.Intersection]
        connections = [(n.x, n.y) for n in visible if n.type == NodeType.Connection]

        app.intersection_scatter.set_offsets(junctions   or np.empty((0, 2)))
        app.connection_scatter.set_offsets(connections or np.empty((0, 2)))

    # ── Panneaux de signalisation ─────────────────────────────────────

    @staticmethod
    def is_diagonal(a, b):
        return abs(b.y - a.y) != 0

    def draw_speed_signs(self):
        app = self.app
        for patch in app.sign_patches:
            patch.remove()
        app.sign_patches = []
        size = 0.18

        for route in app.routes:
            for i in range(1, len(route) - 1):
                prev_node = route[i - 1]
                curr_node = route[i]
                next_node = route[i + 1]

                inc_diag = self.is_diagonal(prev_node, curr_node)
                out_diag = self.is_diagonal(curr_node, next_node)

                if inc_diag == out_diag:
                    continue

                cx, cy = curr_node.x, curr_node.y

                if not inc_diag and out_diag:
                    # Droit → diagonal : freinage, triangle rouge vers le bas
                    pts   = np.array([[cx-size, cy+size*0.6],
                                      [cx+size, cy+size*0.6],
                                      [cx,      cy-size*0.6]])
                    color = C["red"]
                else:
                    # Diagonal → droit : accélération, triangle vert vers le haut
                    pts   = np.array([[cx-size, cy-size*0.6],
                                      [cx+size, cy-size*0.6],
                                      [cx,      cy+size*0.6]])
                    color = C["green"]

                patch = Polygon(pts, closed=True, facecolor=color,
                                edgecolor=C["white"], linewidth=0.6, alpha=0.90, zorder=13)
                app.ax.add_patch(patch)
                app.sign_patches.append(patch)

    # ── Utilitaire ────────────────────────────────────────────────────

    @staticmethod
    def _interpolate(a, b, t):
        t = max(0.0, min(1.0, float(t)))
        return a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t
