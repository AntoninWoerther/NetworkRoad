import math


class Segment:
    """A straight, directed road segment between two adjacent grid nodes."""

    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.key = (start.id, end.id)
        self.length = math.hypot(end.x - start.x, end.y - start.y)

    def point_at(self, t):
        """Linear interpolation for perfectly straight roads and constant speed."""
        t = max(0.0, min(1.0, float(t)))

        x = self.start.x + (self.end.x - self.start.x) * t
        y = self.start.y + (self.end.y - self.start.y) * t
        return x, y

    def sample(self, progress=1.0):
        """Only two points are needed to draw a straight segment."""
        progress = max(0.0, min(1.0, float(progress)))
        x, y = self.point_at(progress)

        return [self.start.x, x], [self.start.y, y]
