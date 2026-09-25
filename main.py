import matplotlib
matplotlib.use('TkAgg')

import random
from collections import defaultdict

import numpy as np
from matplotlib.animation import FuncAnimation

from config import GRID_SIZE, FPS, MAX_SEED, C
from core.network import NetworkGenerator
from core.traffic import TrafficManager
from Ui.interface import InterfaceBuilder
from Ui.drawing import DrawingManager


class RoadNetworkApp:
    """
    Classe principale — relie la génération du réseau, le trafic,
    l'interface et le dessin. Ne contient que l'état partagé et les callbacks.
    """

    def __init__(self):
        self.cols, self.rows = GRID_SIZE
        self.route_count   = 5
        self.vehicle_count = 10
        self.build_speed   = 1.0
        self.traffic_speed = 3.0

        self.pending_cols        = self.cols
        self.pending_rows        = self.rows
        self.pending_route_count = self.route_count

        self.paused = False
        self.seed   = random.SystemRandom().randint(0, MAX_SEED)
        self.rng    = random.Random(self.seed)

        # État réseau
        self.nodes          = {}
        self.routes         = []
        self.segments       = []
        self.segment_by_key = {}
        self.graph          = defaultdict(list)
        self.node_usage     = defaultdict(int)
        self.start_node     = None
        self.end_node       = None
        self.shortest_path_keys = set()
        self.shortest_distance  = 0.0
        self.route_lengths      = []

        # État construction
        self.built_segment_keys    = set()
        self.build_route_index     = 0
        self.build_edge_index      = 0
        self.build_edge_progress   = 0.0
        self.build_finished        = False
        self.total_build_edges     = 1
        self.completed_build_edges = 0

        self.sign_patches  = []
        self.vehicles      = []
        self.frame_counter = 0

        # Modules
        self.network = NetworkGenerator(self)
        self.traffic = TrafficManager(self)
        self.ui      = InterfaceBuilder(self)
        self.drawing = DrawingManager(self)

        # Initialisation
        self.fig, self.ax = self.ui.create_window()
        self.ui.create_interface()
        self.network.generate_network(seed=self.seed, new_endpoints=True)
        self._rebuild_scene()
        self.sync_seed_box()

        self.animation = FuncAnimation(
            self.fig, self.update,
            interval=1000 / FPS, blit=False, cache_frame_data=False,
        )

    # ── Raccourcis vers les modules ───────────────────────────────────

    def _rebuild_scene(self):
        self.drawing.rebuild_scene()
        self.drawing.reset_build_state()
        self.traffic.make_vehicles()
        self.drawing.update_builder_color()
        self.drawing.update_build_drawing()
        self.update_status_text()
        self.fig.canvas.draw_idle()

    # ── Boucle d'animation ────────────────────────────────────────────

    def update(self, _frame):
        self.frame_counter += 1
        pulse = 220 + 18 * np.sin(self.frame_counter * 0.08)
        self.start_ring.set_sizes([pulse])
        self.end_ring.set_sizes([pulse])

        if self.paused:
            return

        if not self.build_finished:
            self.drawing.advance_build()
            self.drawing.update_build_drawing()
            if self.frame_counter % 3 == 0 or self.build_finished:
                self.update_status_text()
        else:
            self.traffic.update_vehicles()

    # ── Statut ────────────────────────────────────────────────────────

    def update_status_text(self):
        junctions = sum(n.type.name == "Intersection" for n in self.nodes.values())

        if self.paused:
            state = "PAUSED"
        elif self.build_finished:
            state = "TRAFFIC"
        else:
            pct   = min(100, int(100 * self.completed_build_edges / max(self.total_build_edges, 1)))
            arrow = "→" if self.build_route_index % 2 == 0 else "←"
            state = f"BUILD {self.build_route_index + 1}/{self.route_count} {arrow} {pct:02d}%"

        self.header_status.set_text(
            f"SEED {self.seed:09d}   •   {self.route_count} Roads   •   GRID {self.cols}×{self.rows}")

        for key, val in [
            ("state", state), ("seed", f"{self.seed:09d}"), ("grid", f"{self.cols}×{self.rows}"),
            ("segments", len(self.segments)), ("routes", self.route_count),
            ("intersections", junctions), ("vehicles", self.vehicle_count),
            ("shortest", f"{self.shortest_distance:.1f}" if np.isfinite(self.shortest_distance) else "--"),
            ("longest",  f"{max(self.route_lengths):.1f}" if self.route_lengths else "--"),
        ]:
            self.stat_labels[key].set_text(str(val))

    # ── Callbacks boutons ─────────────────────────────────────────────

    def instant_build(self, _=None):
        for seg in self.segments:
            self.built_segment_keys.add(seg.key)
        self.build_finished        = True
        self.completed_build_edges = self.total_build_edges
        self.builder_scatter.set_offsets(np.empty((0, 2)))
        self.builder_glow.set_offsets(np.empty((0, 2)))
        self.drawing.update_build_drawing()
        self.drawing.draw_speed_signs()
        self.update_status_text()
        self.fig.canvas.draw_idle()

    def on_config_change(self, _=None):
        self.pending_route_count = int(self.sliders["Routes"].val)
        self.pending_cols        = int(self.sliders["Columns"].val)
        self.pending_rows        = int(self.sliders["Rows"].val)
        self.seed_feedback.set_text("Configuration changed · press GENERATE")
        self.seed_feedback.set_color(C["orange"])

    def on_vehicle_count(self, value):
        self.vehicle_count = int(value)
        self.traffic.make_vehicles()
        self.update_status_text()
        self.fig.canvas.draw_idle()

    def on_speed_change(self, _=None):
        self.build_speed   = float(self.sliders["Construction"].val)
        self.traffic_speed = float(self.sliders["Traffic"].val)

    def _apply_pending(self):
        self.route_count = self.pending_route_count
        self.cols        = self.pending_cols
        self.rows        = self.pending_rows

    def _regen(self, seed, new_endpoints, msg, color):
        self.seed   = seed
        self.paused = False
        self.pause_button.label.set_text("PAUSE")
        self.network.generate_network(seed=self.seed, new_endpoints=new_endpoints)
        self._rebuild_scene()
        self.sync_seed_box()
        self.seed_feedback.set_text(msg)
        self.seed_feedback.set_color(color)

    def generate_from_controls(self, _=None):
        self._apply_pending()
        self._regen(self.seed, True, "Generated with current seed", C["green"])

    def randomize(self, _=None):
        self._apply_pending()
        self._regen(random.SystemRandom().randint(0, MAX_SEED),
                    True, "New random seed generated", C["green"])

    def new_routes_same_endpoints(self, _=None):
        dims_changed = self.pending_cols != self.cols or self.pending_rows != self.rows
        self.route_count = self.pending_route_count = int(self.sliders["Routes"].val)
        if dims_changed:
            self._apply_pending()
            msg = "Grid changed · new endpoints generated"
        else:
            msg = "New alternating roads · endpoints preserved"
        self._regen(random.SystemRandom().randint(0, MAX_SEED), dims_changed, msg, C["cyan"])

    def replay(self, _=None):
        self.paused = False
        self.pause_button.label.set_text("PAUSE")
        self.drawing.reset_build_state()
        self.traffic.make_vehicles()
        self.drawing.update_builder_color()
        self.drawing.update_build_drawing()
        self.update_status_text()
        self.fig.canvas.draw_idle()

    def toggle_pause(self, _=None):
        self.paused = not self.paused
        self.pause_button.label.set_text("RESUME" if self.paused else "PAUSE")
        self.update_status_text()
        self.fig.canvas.draw_idle()

    def load_seed(self, _=None):
        try:
            seed = int(self.seed_box.text.strip())
            if seed < 0: raise ValueError
        except ValueError:
            self.seed_feedback.set_text("Invalid seed · use a positive integer")
            self.seed_feedback.set_color(C["red"])
            return
        self._apply_pending()
        self._regen(seed % (MAX_SEED + 1), True, "Seed loaded · network reproduced", C["green"])

    def sync_seed_box(self):
        if not hasattr(self, "seed_box"):
            return
        prev = self.seed_box.eventson
        self.seed_box.eventson = False
        self.seed_box.set_val(str(self.seed))
        self.seed_box.eventson = prev

    def show(self):
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    RoadNetworkApp().show()
