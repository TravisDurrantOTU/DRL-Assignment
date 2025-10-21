import pygame
import math
from abc import ABC, abstractmethod

# Game constants
MAX_SPEED = 8
ACCELERATION = 4
FRICTION = 2
TURN_SPEED = 400

TRACK_WIDTH = 200
OUTER_RADIUS_X = 500
OUTER_RADIUS_Y = 350
INNER_RADIUS_X = OUTER_RADIUS_X - TRACK_WIDTH
INNER_RADIUS_Y = OUTER_RADIUS_Y - TRACK_WIDTH

WINDOW_WIDTH = 1080
WINDOW_HEIGHT = 720


class GameObject(ABC):
    @abstractmethod
    def get_render_data(self):
        """Return data needed for rendering this object"""
        pass


class Moveable(GameObject):
    @abstractmethod
    def move(self):
        pass


class Interactable(GameObject):
    @abstractmethod
    def detectCollision(self, car):
        pass


class Checkpoint:
    """Represents a checkpoint that cars must cross in sequence"""
    
    def __init__(self, p1, p2, is_start=False):
        self.p1 = p1
        self.p2 = p2
        self.is_start = is_start

    def get_render_data(self):
        """Return checkpoint rendering data"""
        return {
            'type': 'checkpoint',
            'p1': self.p1,
            'p2': self.p2,
            'is_start': self.is_start
        }

    def detectCollision(self, car):
        """Check if car's movement crossed this checkpoint"""
        return self.check_cross((car.old_x, car.old_y), (car.x, car.y))

    def check_cross(self, old_pos, new_pos):
        """Check if line segment old_pos->new_pos crosses this checkpoint"""
        x1, y1 = self.p1
        x2, y2 = self.p2
        x3, y3 = old_pos
        x4, y4 = new_pos

        denom = (x4 - x3) * (y2 - y1) - (y4 - y3) * (x2 - x1)
        if abs(denom) < 1e-10:
            return False

        t = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / denom
        u = -((x1 - x3) * (y4 - y3) - (y1 - y3) * (x4 - x3)) / denom

        return 0 <= t <= 1 and 0 <= u <= 1
    
    def distance_to_point(self, point):
        """Calculate minimum distance from point to this checkpoint line"""
        x0, y0 = point
        x1, y1 = self.p1
        x2, y2 = self.p2
        
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return math.hypot(x0 - x1, y0 - y1)
        
        t = max(0, min(1, ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)))
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        return math.hypot(x0 - closest_x, y0 - closest_y)


class Track(GameObject):
    """Oval racing track with checkpoints and collision detection"""
    
    def __init__(self, center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)):
        self.center = center
        self.outer_rect = pygame.Rect(
            center[0] - OUTER_RADIUS_X,
            center[1] - OUTER_RADIUS_Y,
            OUTER_RADIUS_X * 2,
            OUTER_RADIUS_Y * 2,
        )
        self.inner_rect = pygame.Rect(
            center[0] - INNER_RADIUS_X,
            center[1] - INNER_RADIUS_Y,
            INNER_RADIUS_X * 2,
            INNER_RADIUS_Y * 2,
        )

        self.checkpoints = self._construct_checkpoints()
        self.next_checkpoint_index = 0
        self.laps_completed = 0
        self.checkpoint_states = [False] * len(self.checkpoints)
        self.checkpoint_states[0] = True  # Start already crossed

    def get_render_data(self):
        """Return track rendering data"""
        return {
            'type': 'oval_track',
            'outer_rect': self.outer_rect,
            'inner_rect': self.inner_rect,
            'checkpoints': self.checkpoints,
            'next_checkpoint_index': self.next_checkpoint_index
        }

    def _construct_checkpoints(self):
        """Create 4 checkpoints around the oval track"""
        cx, cy = self.center
        outer_rx, outer_ry = OUTER_RADIUS_X, OUTER_RADIUS_Y
        inner_rx, inner_ry = INNER_RADIUS_X, INNER_RADIUS_Y

        mid_rx = (outer_rx + inner_rx) / 2
        mid_ry = (outer_ry + inner_ry) / 2
        half_thickness_x = (outer_rx - inner_rx) / 2
        half_thickness_y = (outer_ry - inner_ry) / 2

        checkpoints = []
        # Start at bottom (270°), then right, top, left
        for i, angle_deg in enumerate([270, 0, 90, 180]):
            angle = math.radians(angle_deg)
            dx = math.cos(angle)
            dy = math.sin(angle)

            p1 = (
                cx + dx * (mid_rx - half_thickness_x - 10),
                cy + dy * (mid_ry - half_thickness_y - 10),
            )
            p2 = (
                cx + dx * (mid_rx + half_thickness_x + 10),
                cy + dy * (mid_ry + half_thickness_y + 10),
            )

            checkpoints.append(Checkpoint(p1, p2, is_start=(i == 0)))
        return checkpoints

    def check_off_track(self, car_x, car_y):
        """Check if car position is off the track"""
        dx = car_x - self.center[0]
        dy = car_y - self.center[1]

        outer_val = (dx ** 2) / (OUTER_RADIUS_X ** 2) + (dy ** 2) / (OUTER_RADIUS_Y ** 2)
        inner_val = (dx ** 2) / (INNER_RADIUS_X ** 2) + (dy ** 2) / (INNER_RADIUS_Y ** 2)

        return outer_val > 1 or inner_val < 1

    def check_checkpoint_crossing(self, car):
        """Check if car crossed the next checkpoint in sequence"""
        if not self.checkpoints:
            return

        next_checkpoint = self.checkpoints[self.next_checkpoint_index]
        
        # Primary: line intersection
        crossed = next_checkpoint.detectCollision(car)
        
        # Fallback: proximity check
        if not crossed:
            dist = next_checkpoint.distance_to_point((car.x, car.y))
            if dist < 25:
                crossed = True
        
        if crossed:
            self.checkpoint_states[self.next_checkpoint_index] = True
            self.next_checkpoint_index = (self.next_checkpoint_index + 1) % len(self.checkpoints)

            # Lap completed when returning to start after all checkpoints
            if self.next_checkpoint_index == 0:
                if all(self.checkpoint_states):
                    self.laps_completed += 1
                
                # Reset for next lap
                self.checkpoint_states = [False] * len(self.checkpoints)
                self.checkpoint_states[0] = True

    def reset(self):
        """Reset track state"""
        self.next_checkpoint_index = 0
        self.laps_completed = 0
        self.checkpoint_states = [False] * len(self.checkpoints)
        self.checkpoint_states[0] = True


class PolygonTrack:
    """Racing track generated around an arbitrary polygon"""

    def __init__(self, polygon_points, track_width=80, smoothing=0, 
                 num_checkpoints=None, window_size=(800, 600)):
        """
        Initialize a polygon track.
        
        Args:
            polygon_points: List of (x, y) tuples defining the polygon vertices
            track_width: Width of the track in pixels
            smoothing: Number of interpolation points between vertices (0 = sharp corners)
            num_checkpoints: Number of checkpoints (None = auto, one per side)
            window_size: (width, height) of the game window
        """
        self.polygon_points = polygon_points
        self.track_width = track_width
        self.smoothing = smoothing
        self.window_size = window_size
        
        # Generate the track geometry
        self.centerline = self._generate_centerline()
        self.inner_edge, self.outer_edge = self._generate_track_edges()
        
        # Create checkpoints
        if num_checkpoints is None:
            num_checkpoints = len(polygon_points)
        self.checkpoints = self._construct_checkpoints(num_checkpoints)
        self.next_checkpoint_index = 0
        self.laps_completed = 0
        self.checkpoint_states = [False] * len(self.checkpoints)
        self.checkpoint_states[0] = True  # Start already crossed

    def get_render_data(self):
        """Return track rendering data"""
        return {
            'type': 'polygon_track',
            'inner_edge': self.inner_edge,
            'outer_edge': self.outer_edge,
            'checkpoints': self.checkpoints,
            'next_checkpoint_index': self.next_checkpoint_index,
            'track_width': self.track_width
        }

    def _generate_centerline(self):
        """Generate centerline from polygon points with optional smoothing"""
        if self.smoothing == 0:
            return self.polygon_points
        
        # Smooth using Catmull-Rom spline interpolation
        centerline = []
        n = len(self.polygon_points)
        
        for i in range(n):
            p0 = self.polygon_points[(i - 1) % n]
            p1 = self.polygon_points[i]
            p2 = self.polygon_points[(i + 1) % n]
            p3 = self.polygon_points[(i + 2) % n]
            
            for j in range(self.smoothing):
                t = j / self.smoothing
                t2 = t * t
                t3 = t2 * t
                
                x = 0.5 * (
                    (2 * p1[0]) +
                    (-p0[0] + p2[0]) * t +
                    (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                    (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
                )
                y = 0.5 * (
                    (2 * p1[1]) +
                    (-p0[1] + p2[1]) * t +
                    (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                    (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
                )
                
                centerline.append((x, y))
        
        return centerline

    def _generate_track_edges(self):
        """Generate inner and outer track edges using miter method for consistent width"""
        inner_edge = []
        outer_edge = []
        half_width = self.track_width / 2
        
        n = len(self.centerline)
        
        for i in range(n):
            prev_idx = (i - 1) % n
            curr_idx = i
            next_idx = (i + 1) % n
            
            p_prev = self.centerline[prev_idx]
            p_curr = self.centerline[curr_idx]
            p_next = self.centerline[next_idx]
            
            v1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
            v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])
            
            len1 = math.hypot(v1[0], v1[1])
            len2 = math.hypot(v2[0], v2[1])
            
            if len1 > 0:
                v1 = (v1[0] / len1, v1[1] / len1)
            else:
                v1 = (1, 0)
                
            if len2 > 0:
                v2 = (v2[0] / len2, v2[1] / len2)
            else:
                v2 = (1, 0)
            
            n1 = (-v1[1], v1[0])
            n2 = (-v2[1], v2[0])
            
            miter = (n1[0] + n2[0], n1[1] + n2[1])
            miter_len = math.hypot(miter[0], miter[1])
            
            if miter_len > 0:
                miter = (miter[0] / miter_len, miter[1] / miter_len)
            else:
                miter = n1
            
            dot_product = n1[0] * miter[0] + n1[1] * miter[1]
            
            if abs(dot_product) < 0.3:
                miter_scale = half_width / 0.3
                miter_scale = min(miter_scale, half_width * 3)
            else:
                miter_scale = half_width / dot_product
                miter_scale = max(min(miter_scale, half_width * 3), half_width * 0.5)
            
            offset_x = miter[0] * miter_scale
            offset_y = miter[1] * miter_scale
            
            x, y = p_curr
            
            inner_edge.append((x + offset_x, y + offset_y))
            outer_edge.append((x - offset_x, y - offset_y))
        
        return inner_edge, outer_edge

    def _construct_checkpoints(self, num_checkpoints):
        """Create checkpoints evenly distributed along the track"""
        checkpoints = []
        interval = len(self.centerline) // num_checkpoints
        
        for i in range(num_checkpoints):
            idx = (i * interval) % len(self.centerline)
            p_inner = self.inner_edge[idx]
            p_outer = self.outer_edge[idx]
            checkpoints.append(Checkpoint(p_inner, p_outer, is_start=(i == 0)))
        
        return checkpoints

    def check_off_track(self, car_x, car_y):
        """Check if car position is off the track using point-in-polygon test"""
        if not self._point_in_polygon(car_x, car_y, self.outer_edge):
            return True
        
        if self._point_in_polygon(car_x, car_y, self.inner_edge):
            return True
        
        return False

    def _point_in_polygon(self, x, y, polygon):
        """Ray casting algorithm to test if point is inside polygon"""
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside

    def check_checkpoint_crossing(self, car):
        """Check if car crossed the next checkpoint in sequence"""
        if not self.checkpoints:
            return

        next_checkpoint = self.checkpoints[self.next_checkpoint_index]
        
        crossed = next_checkpoint.detectCollision(car)
        
        if not crossed:
            dist = next_checkpoint.distance_to_point((car.x, car.y))
            if dist < 25:
                crossed = True
        
        if crossed:
            self.checkpoint_states[self.next_checkpoint_index] = True
            self.next_checkpoint_index = (self.next_checkpoint_index + 1) % len(self.checkpoints)

            if self.next_checkpoint_index == 0:
                if all(self.checkpoint_states):
                    self.laps_completed += 1
                
                self.checkpoint_states = [False] * len(self.checkpoints)
                self.checkpoint_states[0] = True

    def reset(self):
        """Reset track state"""
        self.next_checkpoint_index = 0
        self.laps_completed = 0
        self.checkpoint_states = [False] * len(self.checkpoints)
        self.checkpoint_states[0] = True


class Car(Moveable):
    """Base class for cars"""
    
    @abstractmethod
    def update_velocity(self, dt):
        pass


class HumanPlayer(Car):
    """Player-controlled car"""
    
    START_X = 200
    START_Y = 360
    START_ANGLE = 270
    START_SPEED = 0

    COLOR = (255, 0, 0)
    SIZE = (40, 20)

    def __init__(self):
        self.x = self.START_X
        self.y = self.START_Y
        self.old_x = self.x
        self.old_y = self.y
        self.angle = self.START_ANGLE
        self.speed = self.START_SPEED

    def get_render_data(self):
        """Return car rendering data"""
        return {
            'type': 'car',
            'x': self.x,
            'y': self.y,
            'angle': self.angle,
            'color': self.COLOR,
            'size': self.SIZE
        }

    def update_velocity(self, dt):
        """Update velocity based on keyboard input"""
        keys = pygame.key.get_pressed()

        # Acceleration
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.speed += ACCELERATION * dt
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            if self.speed > 0:
                self.speed = 0  # Hard brake
            else:
                self.speed -= ACCELERATION * dt * 0.5
        else:
            # Apply friction
            if self.speed > 0:
                self.speed = max(0, self.speed - FRICTION * dt)
            elif self.speed < 0:
                self.speed = min(0, self.speed + FRICTION * dt)

        # Clamp speed
        self.speed = max(-MAX_SPEED / 2, min(self.speed, MAX_SPEED))

        # Steering (speed-dependent)
        if self.speed != 0:
            turn_factor = self.speed / MAX_SPEED
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                self.angle -= TURN_SPEED * dt * turn_factor
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                self.angle += TURN_SPEED * dt * turn_factor

    def move(self):
        """Move car based on current speed and angle"""
        self.old_x = self.x
        self.old_y = self.y
        
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y += math.sin(rad) * self.speed

    def reset(self):
        """Reset car to starting position"""
        self.x = self.START_X
        self.y = self.START_Y
        self.old_x = self.x
        self.old_y = self.y
        self.angle = self.START_ANGLE
        self.speed = self.START_SPEED


class RacingVisual:
    """Handles all visual rendering for the racing game"""
    
    # Color constants
    TRACK_COLOR = (50, 50, 50)
    BORDER_COLOR = (200, 200, 200)
    GRASS_COLOR = (20, 120, 20)
    
    def __init__(self, window_size=(WINDOW_WIDTH, WINDOW_HEIGHT)):
        pygame.init()
        self.window_size = window_size
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Racing Game")
        
        # Cache for pre-rendered surfaces
        self.track_cache = {}
        self.car_cache = {}
        
        # HUD font
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 24, bold=True)
        self.hud_color = (255, 255, 255)
    
    def render_frame(self, track, car, fps):
        """Render a complete frame"""
        self.screen.fill((255, 255, 255))
        self._render_track(track)
        self._render_car(car)
        self._render_hud(fps, car, track)
        pygame.display.flip()
    
    def _render_track(self, track):
        """Render the track"""
        track_data = track.get_render_data()
        track_id = id(track)
        
        # Check if we have a cached surface
        if track_id not in self.track_cache:
            self.track_cache[track_id] = self._create_track_surface(track_data)
        
        self.screen.blit(self.track_cache[track_id], (0, 0))
        
        # Render checkpoints (dynamic, not cached)
        for i, checkpoint in enumerate(track_data['checkpoints']):
            self._render_checkpoint(checkpoint, i == track_data['next_checkpoint_index'])
    
    def _create_track_surface(self, track_data):
        """Create a cached surface for the track"""
        surface = pygame.Surface(self.window_size)
        surface.fill(self.GRASS_COLOR)
        
        if track_data['type'] == 'oval_track':
            pygame.draw.ellipse(surface, self.BORDER_COLOR, track_data['outer_rect'])
            pygame.draw.ellipse(surface, self.TRACK_COLOR, track_data['outer_rect'].inflate(-10, -10))
            pygame.draw.ellipse(surface, self.GRASS_COLOR, track_data['inner_rect'])
        
        elif track_data['type'] == 'polygon_track':
            outer_edge = track_data['outer_edge']
            inner_edge = track_data['inner_edge']
            
            if len(outer_edge) >= 3 and len(inner_edge) >= 3:
                pygame.draw.polygon(surface, self.BORDER_COLOR, outer_edge, 0)
                pygame.draw.polygon(surface, self.TRACK_COLOR, outer_edge, 0)
                
                border_width = max(3, int(track_data['track_width'] * 0.05))
                pygame.draw.polygon(surface, self.BORDER_COLOR, inner_edge, border_width)
                pygame.draw.polygon(surface, self.GRASS_COLOR, inner_edge, 0)
        
        return surface
    
    def _render_checkpoint(self, checkpoint, is_next):
        """Render a checkpoint line"""
        cp_data = checkpoint.get_render_data()
        
        if is_next:
            color = (0, 255, 0)  # Green for next checkpoint
        elif cp_data['is_start']:
            color = (0, 255, 255)  # Cyan for start/finish
        else:
            color = (255, 255, 0)  # Yellow for others
        
        pygame.draw.line(self.screen, color, cp_data['p1'], cp_data['p2'], 3)
    
    def _render_car(self, car):
        """Render the car"""
        car_data = car.get_render_data()
        car_key = (car_data['size'], car_data['color'])
        
        # Check cache for base car surface
        if car_key not in self.car_cache:
            base_surface = pygame.Surface(car_data['size'], pygame.SRCALPHA)
            base_surface.fill(car_data['color'])
            self.car_cache[car_key] = base_surface
        
        # Rotate and blit
        rotated = pygame.transform.rotate(self.car_cache[car_key], -car_data['angle'])
        rect = rotated.get_rect(center=(car_data['x'], car_data['y']))
        self.screen.blit(rotated, rect.topleft)
        
        # Draw front indicator
        half_length = car_data['size'][0] / 2
        rad = math.radians(car_data['angle'])
        front_x = car_data['x'] + math.cos(rad) * half_length
        front_y = car_data['y'] + math.sin(rad) * half_length
        pygame.draw.circle(self.screen, (0, 0, 255), (int(front_x), int(front_y)), 4)
    
    def _render_hud(self, fps, car, track):
        """Render the heads-up display"""
        lines = [
            f"FPS: {int(fps)}",
            f"Speed: {abs(car.speed):.1f}",
            f"Lap: {track.laps_completed}",
            f"Next CP: {track.next_checkpoint_index + 1}/{len(track.checkpoints)}"
        ]

        y_offset = 10
        for line in lines:
            surface = self.font.render(line, True, self.hud_color)
            self.screen.blit(surface, (10, y_offset))
            y_offset += 30
    
    def clear_cache(self):
        """Clear all cached surfaces"""
        self.track_cache.clear()
        self.car_cache.clear()


class RaceController:
    """Main controller for the racing game"""
    
    def __init__(self, visuals=True):
        self.running = True
        self.visuals = visuals
        self.clock = pygame.time.Clock()
        
        if self.visuals:
            self.renderer = RacingVisual()
        else:
            self.renderer = None
        
        self.player = HumanPlayer()

        polygon_points = [
            (180, 280), (220, 200), (300, 160), (420, 160),
            (520, 200), (580, 280), (620, 360), (700, 400),
            (820, 420), (880, 480), (860, 560), (760, 600),
            (620, 600), (500, 580), (400, 540), (320, 480),
            (260, 420), (200, 360), (180, 320)
        ]

        self.track = PolygonTrack(
            polygon_points=polygon_points,
            track_width=100,
            smoothing=15,
            window_size=(WINDOW_WIDTH, WINDOW_HEIGHT)
        )

    def run(self):
        """Main game loop"""
        while self.running:
            dt = self.clock.tick(60) / 1000

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.player.reset()
                        self.track.reset()

            # Update
            self.player.update_velocity(dt)
            self.player.move()
            
            # Check collisions
            self.track.check_checkpoint_crossing(self.player)
            
            if self.track.check_off_track(self.player.x, self.player.y):
                self.player.reset()
                self.track.reset()

            # Render
            if self.visuals and self.renderer:
                self.renderer.render_frame(self.track, self.player, self.clock.get_fps())

        pygame.quit()


if __name__ == "__main__":
    controller = RaceController(visuals=True)
    controller.run()