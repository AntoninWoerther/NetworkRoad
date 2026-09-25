import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox

from config import C, GRID_SIZE


class InterfaceBuilder:
    """Crée et configure tous les éléments visuels de l'interface (fenêtre, panneaux, sliders, boutons)."""

    def __init__(self, app):
        self.app = app

    # ── Fenêtre ───────────────────────────────────────────────────────

    def create_window(self):
        app = self.app
        plt.rcParams["toolbar"] = "None"
        fig = plt.figure(figsize=(15, 8.5), facecolor=C["bg"])
        try:
            fig.canvas.manager.set_window_title("RoadNetwork")
        except Exception:
            pass

        ax = fig.add_axes([0.045, 0.245, 0.705, 0.605])
        ax.set_facecolor(C["bg"])
        ax.set_xlim(-0.9, app.cols - 0.1)
        ax.set_ylim(-0.9, app.rows - 0.1)
        ax.set_aspect("equal")
        ax.axis("off")
        return fig, ax

    # ── Interface complète ────────────────────────────────────────────

    def create_interface(self):
        app = self.app
        app.fig.text(0.05, 0.955, "ROAD NETWORK",
                     color=C["white"], fontsize=24, fontweight="bold")
        app.fig.text(0.05, 0.922,
                     "Sequential procedural roads  /  two-way traffic  /  deterministic seeds",
                     color=C["muted"], fontsize=10)
        app.header_status = app.fig.text(
            0.05, 0.882, "", color=C["cyan"], fontsize=9.5, family="monospace",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": C["panel"],
                  "edgecolor": C["border"], "linewidth": 1})

        self._create_side_panel()
        self._create_controls()

    def _create_side_panel(self):
        app = self.app
        ax  = app.fig.add_axes([0.775, 0.245, 0.195, 0.605])
        ax.set_facecolor(C["panel"])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(C["border"]); spine.set_linewidth(1.1)
        app.side_ax = ax

        ax.text(0.09, 0.94,  "NETWORK STATUS", color=C["white"], fontsize=12, fontweight="bold", va="top")
        ax.text(0.09, 0.895, "LIVE SIMULATION", color=C["cyan"],  fontsize=7.5, family="monospace", va="top")
        ax.plot([0.09, 0.91], [0.855, 0.855], color=C["border"], linewidth=1)

        app.stat_labels = {}
        y = 0.805
        for key, label in [("state","State"), ("seed","Seed"), ("grid","Grid"),
                            ("segments","Segments"), ("routes","Roads"),
                            ("intersections","Junctions"), ("vehicles","Vehicles"),
                            ("shortest","Shortest"), ("longest","Longest")]:
            ax.text(0.09, y, label.upper(), color=C["muted"], fontsize=7.8, family="monospace", va="center")
            app.stat_labels[key] = ax.text(0.91, y, "", color=C["white"], fontsize=8.9,
                                           family="monospace", ha="right", va="center")
            y -= 0.044

        ax.plot([0.09, 0.91], [0.385, 0.385], color=C["border"], linewidth=1)
        ax.text(0.09, 0.345, "LEGEND", color=C["white"], fontsize=9.5, fontweight="bold")

        y = 0.303
        for color, label in [
            (C["green"],  "Start"),          (C["red"],    "End"),
            (C["orange"], "Junction"),        (C["road"],   "Road"),
            (C["cyan"],   "Shortest path"),   (C["cyan"],   "Traffic START → END"),
            (C["purple"], "Traffic END → START"),
            (C["green"],  "▲ Acceleration zone"),
            (C["red"],    "▼ Braking zone"),
        ]:
            ax.scatter([0.13], [y], s=52, color=color, zorder=2)
            ax.text(0.23, y, label, color=C["white"], fontsize=7.5, va="center")
            y -= 0.033

    def _create_controls(self):
        app = self.app
        app.fig.text(0.05, 0.205, "CONFIGURATION", color=C["muted"], fontsize=8, family="monospace")

        app.sliders = {}
        for name, bounds, label, mn, mx, val, step in [
            ("Routes",       [0.075, 0.155, 0.18, 0.019], "Roads",         1,    9,   app.route_count,   1),
            ("Vehicles",     [0.345, 0.155, 0.18, 0.019], "Vehicles",      2,    30,  app.vehicle_count,  1),
            ("Columns",      [0.075, 0.110, 0.18, 0.019], "Columns",       12,   36,  app.cols,           1),
            ("Rows",         [0.345, 0.110, 0.18, 0.019], "Rows",          8,    20,  app.rows,           1),
            ("Construction", [0.075, 0.065, 0.18, 0.019], "spd generation",0.25, 4.0, app.build_speed,   None),
            ("Traffic",      [0.345, 0.065, 0.18, 0.019], "spd vehicles",  0.25, 8.0, app.traffic_speed, None),
        ]:
            app.sliders[name] = self._make_slider(bounds, label, mn, mx, val, step)

        app.sliders["Routes"].on_changed(app.on_config_change)
        app.sliders["Columns"].on_changed(app.on_config_change)
        app.sliders["Rows"].on_changed(app.on_config_change)
        app.sliders["Vehicles"].on_changed(app.on_vehicle_count)
        app.sliders["Construction"].on_changed(app.on_speed_change)
        app.sliders["Traffic"].on_changed(app.on_speed_change)

        app.generate_button  = self._make_button([0.58,  0.152, 0.085, 0.044], "GENERATE")
        app.instant_button   = self._make_button([0.674, 0.152, 0.085, 0.044], "INSTANT")
        app.random_button    = self._make_button([0.768, 0.152, 0.075, 0.044], "RANDOMIZE")
        app.pause_button     = self._make_button([0.852, 0.152, 0.060, 0.044], "PAUSE")
        app.replay_button    = self._make_button([0.921, 0.152, 0.055, 0.044], "REPLAY")
        app.new_path_button  = self._make_button([0.58,  0.098, 0.19,  0.038], "NEW Roads · SAME ENDS")

        seed_ax = app.fig.add_axes([0.79, 0.098, 0.105, 0.038])
        seed_ax.set_facecolor(C["panel"])
        app.seed_box = TextBox(seed_ax, "", initial=str(app.seed),
                               color=C["panel"], hovercolor=C["panel_2"])
        app.seed_box.text_disp.set_color(C["white"])
        app.seed_box.text_disp.set_fontfamily("monospace")
        app.seed_box.text_disp.set_fontsize(9)
        for spine in seed_ax.spines.values():
            spine.set_color(C["border"])

        app.load_seed_button = self._make_button([0.905, 0.098, 0.055, 0.038], "LOAD")
        app.fig.text(0.79, 0.139, "SEED", color=C["muted"], fontsize=7.3, family="monospace")
        app.seed_feedback = app.fig.text(
            0.58, 0.058, "Change grid/roads, then press GENERATE",
            color=C["muted"], fontsize=7.2)

        app.generate_button.on_clicked(app.generate_from_controls)
        app.instant_button.on_clicked(app.instant_build)
        app.random_button.on_clicked(app.randomize)
        app.pause_button.on_clicked(app.toggle_pause)
        app.replay_button.on_clicked(app.replay)
        app.new_path_button.on_clicked(app.new_routes_same_endpoints)
        app.load_seed_button.on_clicked(app.load_seed)
        app.seed_box.on_submit(app.load_seed)

    def _make_slider(self, bounds, label, mn, mx, val, step):
        ax = self.app.fig.add_axes(bounds)
        ax.set_facecolor(C["panel"])
        s = Slider(ax, label, mn, mx, valinit=val, valstep=step, color=C["cyan"])
        s.label.set_color(C["white"]);   s.label.set_fontsize(8.5)
        s.valtext.set_color(C["white"]); s.valtext.set_fontsize(8.5)
        return s

    def _make_button(self, bounds, label):
        ax = self.app.fig.add_axes(bounds)
        ax.set_zorder(50)
        for spine in ax.spines.values():
            spine.set_color(C["border"])
        b = Button(ax, label, color=C["panel"], hovercolor=C["panel_2"])
        b.label.set_color(C["white"]); b.label.set_fontsize(8.2)
        return b
