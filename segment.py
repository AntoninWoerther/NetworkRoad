import math


class Segment:
    def __init__(self, start, end, bend=0.0):
        self.start = start
        self.end = end
        self.bend = float(bend)
        self.key = tuple(sorted((start.id, end.id)))
        self.length = math.hypot(end.x - start.x, end.y - start.y)

    def point_at(self, t):
        t = max(0.0, min(1.0, float(t)))

        x1, y1 = self.start.x, self.start.y
        x2, y2 = self.end.x, self.end.y

        dx = x2 - x1
        dy = y2 - y1

        # Smoothstep removes the rigid "teleport" feeling when a vehicle
        # crosses a junction.
        s = t * t * (3.0 - 2.0 * t)

        x = x1 + dx * s
        y = y1 + dy * s

        # Tiny perpendicular curve: roads remain readable but feel organic.
        length = max(math.hypot(dx, dy), 1e-6)
        nx = -dy / length
        ny = dx / length
        arc = math.sin(math.pi * t) * self.bend

        return x + nx * arc, y + ny * arc

    def sample(self, count=32, progress=1.0):
        progress = max(0.0, min(1.0, float(progress)))
        if progress == 0.0:
            x, y = self.point_at(0.0)
            return [x, x], [y, y]

        points = [
            self.point_at(progress * i / max(count - 1, 1))
            for i in range(count)
        ]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return xs, ys
