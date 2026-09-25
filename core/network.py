import heapq
import random
from collections import defaultdict
from functools import lru_cache

import numpy as np

from node import Node, NodeType
from segment import Segment
from config import MAX_SEED


class NetworkGenerator:
    """Gère la génération du réseau : grille, routes, segments, classification des noeuds et chemin le plus court."""

    def __init__(self, app):
        # Référence à l'application principale pour accéder à l'état partagé
        self.app = app

    def make_grid(self):
        self.app.nodes = {}
        for y in range(self.app.rows):
            for x in range(self.app.cols):
                node_id = y * self.app.cols + x
                self.app.nodes[(x, y)] = Node(node_id, x, y)

    @staticmethod
    def segment_key(a, b):
        return tuple(sorted((a.id, b.id)))

    def generate_network(self, seed=None, new_endpoints=True):
        app = self.app
        old_start_y = app.start_node.y if app.start_node is not None else None
        old_end_y   = app.end_node.y   if app.end_node   is not None else None

        if seed is not None:
            app.seed = int(seed) % (MAX_SEED + 1)

        app.rng = random.Random(app.seed)
        self.make_grid()

        app.routes          = []
        app.segments        = []
        app.segment_by_key  = {}
        app.graph           = defaultdict(list)
        app.node_usage      = defaultdict(int)
        app.route_lengths   = []
        app.shortest_path_keys = set()
        app.shortest_distance  = 0.0

        use_old = (
            not new_endpoints
            and old_start_y is not None
            and old_end_y is not None
            and 0 <= old_start_y < app.rows
            and 0 <= old_end_y < app.rows
        )
        start_y = old_start_y if use_old else app.rng.randint(2, app.rows - 3)
        end_y   = old_end_y   if use_old else app.rng.randint(2, app.rows - 3)

        app.start_node = app.nodes[(0, start_y)]
        app.end_node   = app.nodes[(app.cols - 1, end_y)]

        for route_index in range(app.route_count):
            source = app.start_node if route_index % 2 == 0 else app.end_node
            target = app.end_node   if route_index % 2 == 0 else app.start_node

            route = self.generate_route(route_index, source, target)
            app.routes.append(route)

            for node in route:
                app.node_usage[node] += 1

            route_length = 0.0
            for a, b in zip(route, route[1:]):
                segment = self.register_segment(a, b)
                route_length += segment.length
            app.route_lengths.append(route_length)

        app.total_build_edges = max(
            1, sum(max(0, len(r) - 1) for r in app.routes))

        self.classify_nodes()
        self.compute_shortest_path()

    def generate_route(self, route_index, source, target):
        app            = self.app
        direction_sign = 1 if target.x > source.x else -1
        target_x       = target.x
        target_y       = target.y

        delta_y = {"U": 1, "D": -1, "F": 0, "FU": 1, "FD": -1}
        vectors = {
            "U":  (0, 1), "D": (0, -1),
            "F":  (direction_sign, 0),
            "FU": (direction_sign, 1),
            "FD": (direction_sign, -1),
        }

        def angle_is_valid(prev, nxt):
            if prev == "START":
                return True
            ax, ay = vectors[prev]
            bx, by = vectors[nxt]
            return ax * bx + ay * by >= 0

        detour_profile  = 0.0 if app.route_count <= 1 else route_index / (app.route_count - 1)
        vertical_chance = 0.008 + 0.035 * detour_profile
        vertical_patterns = [
            ((),         1.0 - vertical_chance),
            (("U",),     vertical_chance * 0.485),
            (("D",),     vertical_chance * 0.485),
            (("U","U"),  vertical_chance * 0.015),
            (("D","D"),  vertical_chance * 0.015),
        ]
        forward_moves = ("F", "FU", "FD")

        def next_repeat(last, rep, d):
            count = rep + 1 if d == last else 1
            return None if count > 2 else count

        def simulate_action(x, y, last, rep, verticals, forward):
            ly, ll, lr, nodes, directions = y, last, rep, [], []
            for d in verticals:
                if not angle_is_valid(ll, d): return None
                r = next_repeat(ll, lr, d)
                if r is None: return None
                ly += delta_y[d]
                if not 0 <= ly < app.rows: return None
                nodes.append(app.nodes[(x, ly)])
                directions.append(d)
                ll, lr = d, r
            if not angle_is_valid(ll, forward): return None
            r = next_repeat(ll, lr, forward)
            if r is None: return None
            ny = ly + delta_y[forward]
            if not 0 <= ny < app.rows: return None
            nx = x + direction_sign
            nodes.append(app.nodes[(nx, ny)])
            directions.append(forward)
            return nodes, directions, nx, ny, forward, r

        @lru_cache(maxsize=None)
        def can_finish(x, y, last, rep):
            if x == target_x: return y == target_y
            for verticals, _ in vertical_patterns:
                for fwd in forward_moves:
                    result = simulate_action(x, y, last, rep, verticals, fwd)
                    if result is None: continue
                    _, _, nx, ny, nl, nc = result
                    if nx == target_x and ny != target_y: continue
                    if can_finish(nx, ny, nl, nc): return True
            return False

        route = [source]; current = source; last = "START"; rep = 0

        while current.x != target_x:
            candidates, weights = [], []
            for verticals, pw in vertical_patterns:
                for fwd in forward_moves:
                    result = simulate_action(current.x, current.y, last, rep, verticals, fwd)
                    if result is None: continue
                    nodes, directions, nx, ny, nl, nc = result
                    if nx == target_x and ny != target_y: continue
                    if not can_finish(nx, ny, nl, nc): continue

                    forward_from = current if len(nodes) == 1 else nodes[-2]
                    crossing     = self.edge_would_cross(forward_from, nodes[-1])
                    before       = abs(target_y - current.y)
                    after        = abs(target_y - ny)
                    imp          = before - after

                    if fwd == "F":    fw = 1.50
                    elif imp > 0:     fw = 1.75
                    elif imp == 0:    fw = 1.00
                    else:             fw = 0.35 + 0.55 * detour_profile

                    reuse  = sum(app.node_usage.get(n, 0) for n in nodes)
                    weight = pw * fw / (1.0 + reuse * 0.50)
                    if crossing:                     weight *= 0.08
                    if route_index == 0 and verticals: weight *= 0.10

                    candidates.append((nodes, directions, nl, nc))
                    weights.append(max(weight, 1e-9))

            if not candidates:
                raise RuntimeError("No valid route can reach the requested endpoint.")

            idx = app.rng.choices(range(len(candidates)), weights=weights, k=1)[0]
            chosen_nodes, _, last, rep = candidates[idx]
            route.extend(chosen_nodes)
            current = chosen_nodes[-1]

        if current != target:
            raise RuntimeError("Generated route did not finish on its target endpoint.")

        for prev, center, following in zip(route, route[1:], route[2:]):
            inc = (center.x - prev.x, center.y - prev.y)
            out = (following.x - center.x, following.y - center.y)
            if inc[0] * out[0] + inc[1] * out[1] < 0:
                raise RuntimeError("Generated route contains an angle below 90 degrees.")

        return route

    def edge_would_cross(self, a, b):
        if a.x == b.x: return False
        lx, hx = min(a.x, b.x), max(a.x, b.x)
        al = a if a.x == lx else b
        ah = b if b.x == hx else a
        for seg in self.app.segments:
            if seg.start.x == seg.end.x: continue
            slx = min(seg.start.x, seg.end.x)
            shx = max(seg.start.x, seg.end.x)
            if slx != lx or shx != hx: continue
            sl = seg.start if seg.start.x == lx else seg.end
            sh = seg.end   if seg.end.x == hx   else seg.start
            if (al.y - sl.y) * (ah.y - sh.y) < 0: return True
        return False

    def register_segment(self, a, b):
        key = self.segment_key(a, b)
        if key in self.app.segment_by_key:
            return self.app.segment_by_key[key]
        seg = Segment(a, b)
        self.app.segment_by_key[key] = seg
        self.app.segments.append(seg)
        self.app.graph[a].append((b, seg))
        self.app.graph[b].append((a, seg))
        return seg

    def classify_nodes(self):
        app = self.app
        for node in app.nodes.values():
            deg = len(app.graph[node])
            if   node == app.start_node: node.type = NodeType.Start
            elif node == app.end_node:   node.type = NodeType.End
            elif deg >= 3:               node.type = NodeType.Intersection
            elif deg > 0:                node.type = NodeType.Connection
            else:                        node.type = NodeType.Unused

    def compute_shortest_path(self):
        app  = self.app
        app.shortest_path_keys = set()
        dist  = {app.start_node: 0.0}
        prev  = {}
        queue = [(0.0, app.start_node.id, app.start_node)]

        while queue:
            d, _, node = heapq.heappop(queue)
            if d != dist.get(node): continue
            if node == app.end_node: break
            for nb, seg in app.graph[node]:
                c = d + seg.length
                if c < dist.get(nb, float("inf")):
                    dist[nb] = c
                    prev[nb] = (node, seg)
                    heapq.heappush(queue, (c, nb.id, nb))

        app.shortest_distance = dist.get(app.end_node, float("inf"))
        node = app.end_node
        while node in prev:
            parent, seg = prev[node]
            app.shortest_path_keys.add(self.segment_key(seg.start, seg.end))
            node = parent
