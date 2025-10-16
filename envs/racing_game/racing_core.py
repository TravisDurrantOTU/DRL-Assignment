import pygame
import math

# Game constants
MAX_SPEED = 16
ACCELERATION = 16
FRICTION = 8
TURN_SPEED = 400

TRACK_WIDTH = 200
OUTER_RADIUS_X = 500
OUTER_RADIUS_Y = 350
INNER_RADIUS_X = OUTER_RADIUS_X - TRACK_WIDTH
INNER_RADIUS_Y = OUTER_RADIUS_Y - TRACK_WIDTH

W, H = 1080, 720

class Checkpoint:
    def __init__(self, p1, p2, is_start=False):
        self.p1 = p1
        self.p2 = p2
        self.is_start = is_start

    def draw(self, screen, is_next=False):
        if is_next:
            color = (0, 255, 0)  # Green for next checkpoint
        elif self.is_start:
            color = (0, 255, 255)  # Cyan for start/finish
        else:
            color = (255, 255, 0)  # Yellow for others
        pygame.draw.line(screen, color, self.p1, self.p2, 3)

    def check_cross(self, old_pos, new_pos):
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
        x0, y0 = point
        x1, y1 = self.p1
        x2, y2 = self.p2
        
        # Vector from p1 to p2
        dx = x2 - x1
        dy = y2 - y1
        
        # If checkpoint is a point
        if dx == 0 and dy == 0:
            return math.hypot(x0 - x1, y0 - y1)
        
        # Parameter t of closest point on line segment
        t = max(0, min(1, ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)))
        
        # Closest point on line segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        return math.hypot(x0 - closest_x, y0 - closest_y)


class Track:
    TRACK_COLOR = (50, 50, 50)
    BORDER_COLOR = (200, 200, 200)
    GRASS_COLOR = (20, 120, 20)

    def __init__(self, center=(W // 2, H // 2)):
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

        self.checkpoints = self._make_checkpoints()
        self.next_checkpoint = 0
        self.laps = 0
        self.checkpoint_states = [False] * len(self.checkpoints)
        self.checkpoint_states[0] = True  # Start checkpoint is already "crossed"

        # Pre-render track surface
        self.surface = pygame.Surface((W, H))
        self.surface.fill(Track.GRASS_COLOR)
        self._draw_track()

    def _draw_track(self):
        pygame.draw.ellipse(self.surface, Track.BORDER_COLOR, self.outer_rect)
        pygame.draw.ellipse(self.surface, Track.TRACK_COLOR, self.outer_rect.inflate(-10, -10))
        pygame.draw.ellipse(self.surface, Track.GRASS_COLOR, self.inner_rect)

    def draw(self, screen):
        screen.blit(self.surface, (0, 0))

    def check_collision(self, car_x, car_y):
        dx = car_x - self.center[0]
        dy = car_y - self.center[1]

        outer_val = (dx ** 2) / (OUTER_RADIUS_X ** 2) + (dy ** 2) / (OUTER_RADIUS_Y ** 2)
        inner_val = (dx ** 2) / (INNER_RADIUS_X ** 2) + (dy ** 2) / (INNER_RADIUS_Y ** 2)

        return outer_val > 1 or inner_val < 1

    def _make_checkpoints(self):
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

    def check_checkpoints(self, car_x, car_y, old_x, old_y):
        if not self.checkpoints:
            return

        next_cp = self.checkpoints[self.next_checkpoint]
        
        # Primary method: line intersection
        crossed = next_cp.check_cross((old_x, old_y), (car_x, car_y))
        
        # Fallback: proximity check if car is very close to checkpoint
        if not crossed:
            dist = next_cp.distance_to_point((car_x, car_y))
            if dist < 25:  # Within 25 pixels of checkpoint line
                crossed = True
        
        if crossed:
            self.checkpoint_states[self.next_checkpoint] = True
            self.next_checkpoint = (self.next_checkpoint + 1) % len(self.checkpoints)

            # Lap completed when crossing start/finish after all checkpoints
            if self.next_checkpoint == 0:
                # Check if all checkpoints were crossed
                if all(self.checkpoint_states):
                    self.laps += 1
                    print(f"Lap {self.laps} completed!")
                # Reset checkpoint states for next lap
                self.checkpoint_states = [False] * len(self.checkpoints)
                self.checkpoint_states[0] = True

    def draw_checkpoints(self, screen):
        for i, cp in enumerate(self.checkpoints):
            cp.draw(screen, is_next=(i == self.next_checkpoint))

    def reset(self):
        self.next_checkpoint = 0
        self.laps = 0
        self.checkpoint_states = [False] * len(self.checkpoints)
        self.checkpoint_states[0] = True


class HumanPlayer:
    START_X = 150
    START_Y = 420
    START_ANGLE = 270
    START_SPEED = 0

    COLOR = (255, 0, 0)
    SIZE = (40, 20)

    def __init__(self):
        self.x = self.START_X
        self.y = self.START_Y
        self.angle = self.START_ANGLE
        self.speed = self.START_SPEED
        
        # Pre-create car surface
        self.base_surface = pygame.Surface(self.SIZE, pygame.SRCALPHA)
        self.base_surface.fill(self.COLOR)

    def draw(self, screen):
        rotated = pygame.transform.rotate(self.base_surface, -self.angle)
        rect = rotated.get_rect(center=(self.x, self.y))
        screen.blit(rotated, rect.topleft)

        # Draw front indicator
        half_length = self.SIZE[0] / 2
        rad = math.radians(self.angle)
        front_x = self.x + math.cos(rad) * half_length
        front_y = self.y + math.sin(rad) * half_length
        pygame.draw.circle(screen, (0, 0, 255), (int(front_x), int(front_y)), 4)

    def update(self, dt):
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
            # Friction
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

        # Movement
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y += math.sin(rad) * self.speed

    def reset(self):
        self.x = self.START_X
        self.y = self.START_Y
        self.angle = self.START_ANGLE
        self.speed = self.START_SPEED


class HUD:
    def __init__(self, font_size=24, color=(255, 255, 255)):
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", font_size, bold=True)
        self.color = color

    def draw(self, screen, fps, car, track):
        lines = [
            f"FPS: {int(fps)}",
            f"Speed: {abs(car.speed):.1f}",
            f"Lap: {track.laps}",
            f"Next CP: {track.next_checkpoint + 1}/{len(track.checkpoints)}"
        ]

        y_offset = 10
        for line in lines:
            surface = self.font.render(line, True, self.color)
            screen.blit(surface, (10, y_offset))
            y_offset += 30


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Racing Game")
    clock = pygame.time.Clock()

    car = HumanPlayer()
    track = Track()
    hud = HUD()

    running = True
    while running:
        dt = clock.tick(60) / 1000
        old_x, old_y = car.x, car.y

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Reset on 'R'
                    car.reset()
                    track.reset()

        car.update(dt)
        track.check_checkpoints(car.x, car.y, old_x, old_y)

        # Collision detection and reset
        if track.check_collision(car.x, car.y):
            car.reset()
            track.reset()

        # Rendering
        screen.fill((255, 255, 255))
        track.draw(screen)
        track.draw_checkpoints(screen)
        car.draw(screen)
        hud.draw(screen, clock.get_fps(), car, track)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()