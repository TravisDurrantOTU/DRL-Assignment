# TODO: Make it so that laps actually count

import pygame
import math
running = True

w,h = 1080, 720
screen = pygame.display.set_mode((w,h))
pygame.display.set_caption("Racing Game")

# game constants
MAX_SPEED = 16
ACCELERATION = 16
FRICTION = 8
TURN_SPEED = 400

TRACK_WIDTH = 200
OUTER_RADIUS_X = 500
OUTER_RADIUS_Y = 350
INNER_RADIUS_X = OUTER_RADIUS_X - TRACK_WIDTH
INNER_RADIUS_Y = OUTER_RADIUS_Y - TRACK_WIDTH

class Checkpoint:
    def __init__(self, p1, p2, is_start=False):
        self.p1 = p1
        self.p2 = p2
        self.is_start = is_start
        self.crossed = False

    def draw(self, screen):
        color = (255, 255, 0) if not self.is_start else (0, 255, 255)
        pygame.draw.line(screen, color, self.p1, self.p2, 3)

    def check_cross(self, old_pos, new_pos):
        x1, y1 = self.p1
        x2, y2 = self.p2
        x3, y3 = old_pos
        x4, y4 = new_pos

        denom = (x4 - x3) * (y2 - y1) - (y4 - y3) * (x2 - x1)
        if denom == 0:
            return False 

        t = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / denom
        u = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / denom

        return 0 <= t <= 1 and 0 <= u <= 1

class Track:
    TRACK_COLOR = (50, 50, 50)
    BORDER_COLOR = (200, 200, 200)
    GRASS_COLOR = (20, 120, 20)

    def __init__(self, center=(w // 2, h // 2)):
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

        # Pre-render track surface for performance
        self.surface = pygame.Surface((w, h))
        self.surface.fill(Track.GRASS_COLOR)
        self._draw_track()

    def _draw_track(self):
        pygame.draw.ellipse(self.surface, Track.BORDER_COLOR, self.outer_rect)
        pygame.draw.ellipse(self.surface, Track.TRACK_COLOR, self.outer_rect.inflate(-10, -10))
        pygame.draw.ellipse(self.surface, Track.GRASS_COLOR, self.inner_rect)

    def draw(self, screen):
        screen.blit(self.surface, (0, 0))

    def check_collision(self, car):
        # Convert to relative position
        dx = car.x - self.center[0]
        dy = car.y - self.center[1]

        # Ellipse equation: (x/a)^2 + (y/b)^2 <= 1
        outer_val = (dx ** 2) / (OUTER_RADIUS_X ** 2) + (dy ** 2) / (OUTER_RADIUS_Y ** 2)
        inner_val = (dx ** 2) / (INNER_RADIUS_X ** 2) + (dy ** 2) / (INNER_RADIUS_Y ** 2)

        # If outside outer ellipse or inside inner ellipse → off track
        if outer_val > 1 or inner_val < 1:
            return True
        return False
    
    # hardcoded for the oval right now
    def _make_checkpoints(self):
        cx, cy = self.center
        outer_rx, outer_ry = OUTER_RADIUS_X, OUTER_RADIUS_Y
        inner_rx, inner_ry = INNER_RADIUS_X, INNER_RADIUS_Y

        # midpoint ellipse between outer and inner
        mid_rx = (outer_rx + inner_rx) / 2
        mid_ry = (outer_ry + inner_ry) / 2

        # the checkpoints will stretch across the track width
        half_thickness_x = (outer_rx - inner_rx) / 2
        half_thickness_y = (outer_ry - inner_ry) / 2

        checkpoints = []
        for i, angle_deg in enumerate([270, 0, 90, 180]):  # bottom, right, top, left
            angle = math.radians(angle_deg)
            dx = math.cos(angle)
            dy = math.sin(angle)

            # long enough to fully span track width + margin
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

    def check_checkpoints(self, car, old_x, old_y):
        if not self.checkpoints:
            return

        # Check all checkpoints, but only accept the next expected one
        next_cp = self.checkpoints[self.next_checkpoint]
        crossed = next_cp.check_cross((old_x, old_y), (car.x, car.y))

        # As backup, check proximity if line intersection missed
        if not crossed:
            dist = math.hypot(car.x - next_cp.p1[0], car.y - next_cp.p1[1])
            dist2 = math.hypot(car.x - next_cp.p2[0], car.y - next_cp.p2[1])
            if dist < 30 or dist2 < 30:  # within 30px margin
                crossed = True

        if crossed:
            # Move to the next checkpoint
            self.next_checkpoint = (self.next_checkpoint + 1) % len(self.checkpoints)

            # If we wrapped back to start, increment lap
            if self.next_checkpoint == 0:
                self.laps += 1
                print(f"Lap {self.laps} completed!")


    def reset_position(self, car):
        car.x = HumanPlayer.START_X
        car.y = HumanPlayer.START_Y
        car.angle = HumanPlayer.START_ANGLE
        car.speed = HumanPlayer.START_SPEED
    

class HumanPlayer():
    START_X = 540
    START_Y = 600
    START_ANGLE = 0
    START_SPEED = 0

    COLOR = (255, 0, 0)
    SIZE = (40, 20)

    def __init__(self):
        self.x = HumanPlayer.START_X
        self.y = HumanPlayer.START_Y
        self.angle = HumanPlayer.START_ANGLE
        self.speed = HumanPlayer.START_SPEED

    def draw(self, screen):
        car_surface = pygame.Surface(HumanPlayer.SIZE)
        car_surface.fill(HumanPlayer.COLOR)
        car_surface.set_colorkey((0, 0, 0))

        rotated = pygame.transform.rotate(car_surface, -self.angle)
        rect = rotated.get_rect(center=(self.x, self.y))
        screen.blit(rotated, rect.topleft)

        half_length = HumanPlayer.SIZE[0] / 2
        rad = math.radians(self.angle)
        front_x = self.x + math.cos(rad) * half_length
        front_y = self.y + math.sin(rad) * half_length
        pygame.draw.circle(screen, (0, 0, 255), (int(front_x), int(front_y)), 4)


    def update_velocity(self, dt=1):
        keys = pygame.key.get_pressed()

        # Forward / backward
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.speed += ACCELERATION * dt
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            # letting the car HARD brake so hopefully some drifting can happen?
            if self.speed > 0:
                self.speed = 0
            else:
                self.speed -= ACCELERATION * dt * 0.5
        else:
            # Apply friction when no input
            if self.speed > 0:
                self.speed -= FRICTION * dt
                self.speed = max(self.speed, 0)
            elif self.speed < 0:
                self.speed += FRICTION * dt
                self.speed = min(self.speed, 0)

        # Clamp speed
        self.speed = max(-MAX_SPEED / 2, min(self.speed, MAX_SPEED))

        # Steering
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.angle -= TURN_SPEED * dt * (self.speed / MAX_SPEED)
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.angle += TURN_SPEED * dt * (self.speed / MAX_SPEED)

    def move(self):
        # Move based on current speed and direction
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y += math.sin(rad) * self.speed

class HUD:
    def __init__(self, font_size=24, color=(255, 255, 255)):
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", font_size)
        self.color = color

    def draw(self, screen, clock, car, track=None):
        # FPS
        fps = int(clock.get_fps())

        # Speed (convert to positive, round)
        speed = abs(car.speed)
        speed_text = f"Speed: {speed:.1f}"

        # lap info
        lap_text = ""
        if track and hasattr(track, "laps"):
            lap_text = f"Lap: {track.laps}"

        # Create text surfaces
        fps_surface = self.font.render(f"FPS: {fps}", True, self.color)
        speed_surface = self.font.render(speed_text, True, self.color)
        lap_surface = self.font.render(lap_text, True, self.color) if lap_text else None

        # Blit them on screen (top-left corner)
        screen.blit(fps_surface, (10, 10))
        screen.blit(speed_surface, (10, 40))
        if lap_surface:
            screen.blit(lap_surface, (10, 70))



car = HumanPlayer()
track = Track()
clock = pygame.time.Clock()
hud = HUD()

while running:

    dt = clock.tick(60) / 1000
    old_x, old_y = car.x, car.y

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill((255, 255, 255))
    car.update_velocity(dt=dt)
    car.move()

    track.check_checkpoints(car, old_x, old_y)

    if track.check_collision(car):
        track.reset_position(car)
    track.draw(screen)
    for checkpoint in track.checkpoints:
        checkpoint.draw(screen)
    car.draw(screen)
    hud.draw(screen, clock, car, track)
    pygame.display.flip()
    clock.tick(60)
pygame.quit()
    