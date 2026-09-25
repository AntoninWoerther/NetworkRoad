import numpy as np
from config import SPEED_MIN, SPEED_MAX, SPEED_LERP


class TrafficManager:
    """Gère la création, le déplacement et la vitesse des véhicules."""

    def __init__(self, app):
        self.app = app

    @staticmethod
    def segment_speed_limit(a, b):
        """
        Segment horizontal (dy == 0) → ligne droite → vitesse max.
        Segment diagonal   (dy != 0) → virage       → vitesse min.
        """
        return SPEED_MIN if abs(b.y - a.y) != 0 else SPEED_MAX

    def route_for_traffic_direction(self, route_index, direction):
        app   = self.app
        route = app.routes[route_index]
        desired_start = app.start_node if direction == 1 else app.end_node
        return route if route[0] == desired_start else list(reversed(route))

    def make_vehicles(self):
        app = self.app
        app.vehicles = []
        if not app.routes:
            return

        for index in range(app.vehicle_count):
            direction   = 1 if index % 2 == 0 else -1
            route_index = index % len(app.routes)
            path        = self.route_for_traffic_direction(route_index, direction)
            if len(path) < 2:
                continue

            app.vehicles.append({
                "direction":    direction,
                "route_index":  route_index,
                "path":         path,
                "edge_index":   0,
                "progress":     (index / max(app.vehicle_count, 1)) * 0.85,
                "speed":        app.rng.uniform(SPEED_MIN, SPEED_MAX),
                "target_speed": SPEED_MAX,
            })

    def restart_vehicle(self, vehicle):
        app = self.app
        vehicle["route_index"] = app.rng.randrange(len(app.routes))
        vehicle["path"]        = self.route_for_traffic_direction(
            vehicle["route_index"], vehicle["direction"])
        vehicle["edge_index"]  = 0
        vehicle["progress"]    = 0.0

    def update_vehicles(self):
        app = self.app
        if not app.build_finished:
            empty = np.empty((0, 2))
            for sc in (app.vehicle_out, app.vehicle_out_glow,
                       app.vehicle_back, app.vehicle_back_glow):
                sc.set_offsets(empty)
            return

        outbound, inbound = [], []

        for vehicle in app.vehicles:
            path       = vehicle["path"]
            edge_index = vehicle["edge_index"]

            if edge_index < len(path) - 1:
                vehicle["target_speed"] = self.segment_speed_limit(
                    path[edge_index], path[edge_index + 1])

            vehicle["speed"] += (vehicle["target_speed"] - vehicle["speed"]) * SPEED_LERP
            remaining = vehicle["speed"] * app.traffic_speed

            while remaining > 0:
                path       = vehicle["path"]
                edge_index = vehicle["edge_index"]

                if edge_index >= len(path) - 1:
                    self.restart_vehicle(vehicle)
                    continue

                a, b   = path[edge_index], path[edge_index + 1]
                length = max(np.hypot(b.x - a.x, b.y - a.y), 1e-9)
                left   = (1.0 - vehicle["progress"]) * length

                if remaining < left:
                    vehicle["progress"] += remaining / length
                    remaining = 0.0
                else:
                    remaining             -= left
                    vehicle["edge_index"] += 1
                    vehicle["progress"]    = 0.0

                    if vehicle["edge_index"] >= len(path) - 1:
                        self.restart_vehicle(vehicle)
                        break

                    vehicle["target_speed"] = self.segment_speed_limit(
                        path[vehicle["edge_index"]],
                        path[vehicle["edge_index"] + 1])

            path       = vehicle["path"]
            edge_index = min(vehicle["edge_index"], len(path) - 2)
            point      = self._interpolate(path[edge_index], path[edge_index + 1], vehicle["progress"])

            (outbound if vehicle["direction"] == 1 else inbound).append(point)

        out  = outbound if outbound else np.empty((0, 2))
        back = inbound  if inbound  else np.empty((0, 2))
        app.vehicle_out.set_offsets(out);       app.vehicle_out_glow.set_offsets(out)
        app.vehicle_back.set_offsets(back);     app.vehicle_back_glow.set_offsets(back)

    @staticmethod
    def _interpolate(a, b, t):
        t = max(0.0, min(1.0, float(t)))
        return a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t
