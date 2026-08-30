"""Colourful manual Pygame demo for the deterministic GravityEnv.

All simulation and collision logic remains in ``gravity_env.py``.  This file
only gathers keyboard input, calls ``GravityEnv.step()``, and renders the
environment's public state.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
from typing import Any

from gravity_env import GravityEnv

try:
    import pygame
except ImportError:  # Allows non-visual tooling to import this module safely.
    pygame = None  # type: ignore[assignment]


WIDTH = 1200
HEIGHT = 800
FPS = 60
TRAIL_LENGTH = 650

SPACE = (3, 5, 17)
WHITE = (232, 242, 255)
MUTED = (126, 151, 187)
CYAN = (69, 225, 255)
GREEN = (85, 255, 174)
RED = (255, 70, 92)

PLANET_PALETTE = (
    (80, 186, 255),
    (255, 102, 124),
    (255, 193, 73),
    (151, 102, 255),
    (74, 222, 172),
    (235, 92, 188),
)


@dataclass(frozen=True)
class AttemptTrail:
    """A real completed/local rollout rendered as a faint comparison path."""

    points: tuple[tuple[float, float], ...]
    reward: float | None
    success: bool | None
    seed: int | None = None
    policy_version: int | None = None
    generation: int | None = None
    sandbox_id: str | None = None


def load_rollout_trails(path: str | Path) -> list[AttemptTrail]:
    """Load real worker output without fabricating training metadata.

    Accepted JSON shapes are one rollout object, a list of rollout objects, or
    ``{"rollouts": [...]}``.  Invalid or empty trajectories are rejected so a
    displayed ghost always corresponds to recorded physics.
    """

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    outer_generation = payload.get("generation") if isinstance(payload, dict) else None
    outer_policy_version = payload.get("policy_version") if isinstance(payload, dict) else None
    if isinstance(payload, dict) and "rollouts" in payload:
        payload = payload["rollouts"]
    elif isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("rollout JSON must be an object, a list, or contain a 'rollouts' list")

    attempts: list[AttemptTrail] = []
    for index, rollout in enumerate(payload):
        if not isinstance(rollout, dict):
            raise ValueError(f"rollout {index} must be a JSON object")
        trajectory = rollout.get("trajectory")
        if not isinstance(trajectory, list) or len(trajectory) < 2:
            raise ValueError(f"rollout {index} needs a trajectory with at least two points")

        points: list[tuple[float, float]] = []
        for point_index, point in enumerate(trajectory):
            if not isinstance(point, dict) or "x" not in point or "y" not in point:
                raise ValueError(f"rollout {index} trajectory point {point_index} needs x and y")
            points.append((float(point["x"]), float(point["y"])))

        sandbox_id = rollout.get("sandbox_id")
        reward = rollout.get("reward")
        success = rollout.get("success")
        rollout_seed = rollout.get("seed")
        rollout_policy = rollout.get("policy_version", outer_policy_version)
        rollout_generation = rollout.get("generation", outer_generation)
        attempts.append(
            AttemptTrail(
                points=tuple(points),
                reward=float(reward) if reward is not None else None,
                success=bool(success) if success is not None else None,
                seed=int(rollout_seed) if rollout_seed is not None else None,
                policy_version=int(rollout_policy) if rollout_policy is not None else None,
                generation=(
                    int(rollout_generation) if rollout_generation is not None else None
                ),
                sandbox_id=str(sandbox_id) if sandbox_id is not None else None,
            )
        )
    return attempts


def _xy(value: Any) -> tuple[float, float]:
    """Convert an environment position or velocity to a drawable pair."""

    return float(value[0]), float(value[1])


def _position(entity: dict[str, Any]) -> tuple[float, float]:
    return _xy(entity["position"])


def make_stars(seed: int, count: int = 190) -> list[tuple[int, int, int, int, float, float]]:
    """Build a deterministic star field without changing the game state."""

    rng = random.Random((int(seed) * 1_000_003) ^ 0x5A17F13D)
    stars: list[tuple[int, int, int, int, float, float]] = []
    for _ in range(count):
        size_roll = rng.random()
        radius = 1 if size_roll < 0.78 else 2 if size_roll < 0.97 else 3
        stars.append(
            (
                rng.randrange(WIDTH),
                rng.randrange(HEIGHT),
                radius,
                rng.randint(115, 225),
                rng.random() * math.tau,
                rng.uniform(0.7, 2.0),
            )
        )
    return stars


def make_background(seed: int) -> tuple[Any, list[tuple[int, int, int, int, float, float]]]:
    """Create the static dark-space background for one seeded universe."""

    assert pygame is not None
    background = pygame.Surface((WIDTH, HEIGHT)).convert()
    background.fill(SPACE)

    nebula = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    rng = random.Random((int(seed) * 97_409) ^ 0xC05A05)
    colours = ((29, 47, 116), (82, 27, 112), (14, 78, 100))
    for index in range(8):
        center = (rng.randrange(-120, WIDTH + 120), rng.randrange(-100, HEIGHT + 100))
        radius = rng.randrange(140, 310)
        colour = colours[index % len(colours)]
        for layer in range(6, 0, -1):
            pygame.draw.circle(
                nebula,
                (*colour, 4 + 6 - layer),
                center,
                int(radius * layer / 6),
            )
    background.blit(nebula, (0, 0))
    return background, make_stars(seed)


def draw_stars(screen: Any, stars: list[tuple[int, int, int, int, float, float]], elapsed: float) -> None:
    assert pygame is not None
    for x, y, radius, base, phase, speed in stars:
        brightness = max(80, min(255, int(base + 28 * math.sin(elapsed * speed + phase))))
        colour = (brightness, min(255, brightness + 7), min(255, brightness + 24))
        pygame.draw.circle(screen, colour, (x, y), radius)
        if radius == 3:
            faint = (brightness // 2, brightness // 2, min(255, brightness // 2 + 25))
            pygame.draw.line(screen, faint, (x - 5, y), (x + 5, y))
            pygame.draw.line(screen, faint, (x, y - 5), (x, y + 5))


def draw_trail(screen: Any, trail: deque[tuple[float, float]]) -> None:
    """Draw old positions dimly and recent positions brightly."""

    assert pygame is not None
    if len(trail) < 2:
        return

    points = list(trail)
    count = len(points)
    for index in range(1, count):
        freshness = index / count
        colour = (
            int(7 + 38 * freshness),
            int(22 + 186 * freshness),
            int(45 + 210 * freshness),
        )
        width = 1 if freshness < 0.55 else 2 if freshness < 0.88 else 3
        pygame.draw.line(screen, colour, points[index - 1], points[index], width)


def draw_ghost_trails(
    screen: Any,
    attempts: list[AttemptTrail],
    current_seed: int,
) -> None:
    """Render recorded attempts, with the real best reward as champion."""

    assert pygame is not None
    eligible = [
        attempt
        for attempt in attempts
        if attempt.seed == current_seed and len(attempt.points) >= 2
    ]
    if not eligible:
        return

    scored = [attempt for attempt in eligible if attempt.reward is not None]
    champion = max(scored, key=lambda attempt: float(attempt.reward)) if scored else None
    layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for attempt in eligible:
        is_champion = champion is not None and attempt is champion
        colour = (89, 255, 183) if is_champion else (115, 139, 185)
        alpha = 150 if is_champion else 42
        width = 3 if is_champion else 1
        pygame.draw.lines(layer, (*colour, alpha), False, attempt.points, width)

        if is_champion:
            endpoint = (round(attempt.points[-1][0]), round(attempt.points[-1][1]))
            pygame.draw.circle(layer, (*colour, 190), endpoint, 6, 2)

    screen.blit(layer, (0, 0))


def draw_champion_label(
    screen: Any,
    attempts: list[AttemptTrail],
    current_seed: int,
    label_font: Any,
) -> None:
    assert pygame is not None
    eligible = [
        attempt
        for attempt in attempts
        if attempt.seed == current_seed
        and attempt.reward is not None
        and len(attempt.points) >= 2
    ]
    if not eligible:
        return
    champion = max(eligible, key=lambda attempt: float(attempt.reward))
    endpoint = (round(champion.points[-1][0]), round(champion.points[-1][1]))
    label = label_font.render("CURRENT CHAMPION", True, (115, 255, 195))
    label_x = max(8, min(WIDTH - label.get_width() - 8, endpoint[0] + 10))
    label_y = max(8, min(HEIGHT - label.get_height() - 8, endpoint[1] - 22))
    screen.blit(label, (label_x, label_y))


def draw_speed_streaks(
    screen: Any,
    position: tuple[float, float],
    velocity: tuple[float, float],
) -> None:
    """Add restrained motion cues derived only from the real ship velocity."""

    assert pygame is not None
    speed = math.hypot(*velocity)
    if speed < 60.0:
        return
    direction = (velocity[0] / speed, velocity[1] / speed)
    side = (-direction[1], direction[0])
    length = min(34.0, 5.0 + speed * 0.065)
    alpha = max(25, min(125, int((speed - 60.0) * 0.7)))
    layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for offset, scale in ((-8.0, 0.55), (0.0, 1.0), (8.0, 0.55)):
        start = (
            position[0] + side[0] * offset - direction[0] * 15.0,
            position[1] + side[1] * offset - direction[1] * 15.0,
        )
        end = (
            start[0] - direction[0] * length * scale,
            start[1] - direction[1] * length * scale,
        )
        pygame.draw.line(layer, (92, 222, 255, int(alpha * scale)), start, end, 2)
    screen.blit(layer, (0, 0))


def _gravity_ring(
    layer: Any,
    center: tuple[int, int],
    radius: int,
    colour: tuple[int, int, int],
) -> None:
    assert pygame is not None
    pygame.draw.circle(layer, (*colour, 32), center, radius, 1)
    pygame.draw.circle(layer, (*colour, 17), center, max(2, int(radius * 0.62)), 1)

    rect = pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2)
    for segment in range(12):
        start = segment * math.tau / 12 + 0.05
        pygame.draw.arc(layer, (*colour, 75), rect, start, start + 0.28, 2)


def draw_planets(screen: Any, planets: list[dict[str, Any]], seed: int) -> None:
    """Draw gravity influence first, then bright spherical planet bodies."""

    assert pygame is not None
    effects = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    for index, planet in enumerate(planets):
        x, y = _position(planet)
        center = (round(x), round(y))
        radius = max(3, round(float(planet["radius"])))
        fallback = PLANET_PALETTE[(seed + index * 3) % len(PLANET_PALETTE)]
        colour = tuple(planet.get("colour", planet.get("color", fallback)))
        gravity_radius = max(radius + 2, round(float(planet.get("gravity_radius", radius * 4))))
        _gravity_ring(effects, center, gravity_radius, colour)
        pygame.draw.circle(effects, (*colour, 18), center, max(radius + 18, int(radius * 1.7)))
        pygame.draw.circle(effects, (*colour, 38), center, radius + 10)

    screen.blit(effects, (0, 0))

    for index, planet in enumerate(planets):
        x, y = _position(planet)
        center = (round(x), round(y))
        radius = max(3, round(float(planet["radius"])))
        fallback = PLANET_PALETTE[(seed + index * 3) % len(PLANET_PALETTE)]
        colour = tuple(planet.get("colour", planet.get("color", fallback)))

        pygame.draw.circle(screen, (7, 9, 21), (center[0] + 4, center[1] + 6), radius + 2)
        pygame.draw.circle(screen, colour, center, radius)
        highlight = tuple(min(255, channel + 100) for channel in colour)
        highlight_center = (center[0] - int(radius * 0.27), center[1] - int(radius * 0.30))
        for layer in range(4, 0, -1):
            local_radius = max(2, int(radius * layer / 8))
            blend = layer / 7
            layer_colour = tuple(
                int(colour[channel] * (1 - blend) + highlight[channel] * blend)
                for channel in range(3)
            )
            pygame.draw.circle(screen, layer_colour, highlight_center, local_radius)


def asteroid_points(
    center: tuple[float, float], radius: float, seed: int, index: int
) -> list[tuple[int, int]]:
    rng = random.Random((int(seed) + 31) * 65_537 + index * 8_191)
    count = rng.randint(8, 11)
    points: list[tuple[int, int]] = []
    for point_index in range(count):
        angle = point_index * math.tau / count
        local_radius = radius * rng.uniform(0.76, 1.12)
        points.append(
            (
                round(center[0] + math.cos(angle) * local_radius),
                round(center[1] + math.sin(angle) * local_radius),
            )
        )
    return points


def draw_asteroids(screen: Any, asteroids: list[dict[str, Any]], seed: int) -> None:
    assert pygame is not None
    for index, asteroid in enumerate(asteroids):
        center = _position(asteroid)
        radius = float(asteroid["radius"])
        points = asteroid_points(center, radius, seed, index)
        shadow = [(x + 3, y + 4) for x, y in points]
        base = tuple(asteroid.get("colour", asteroid.get("color", (91, 98, 114))))

        pygame.draw.polygon(screen, (12, 15, 26), shadow)
        pygame.draw.polygon(screen, base, points)
        pygame.draw.lines(screen, (160, 170, 190), True, points, 2)

        x, y = round(center[0]), round(center[1])
        crater = max(2, int(radius * 0.19))
        pygame.draw.circle(
            screen,
            (54, 60, 75),
            (x - int(radius * 0.25), y - int(radius * 0.16)),
            crater,
        )


def draw_portal(screen: Any, portal: dict[str, Any], elapsed: float) -> None:
    assert pygame is not None
    x, y = _position(portal)
    center = (round(x), round(y))
    radius = max(5, round(float(portal["radius"])))
    pulse = 1.0 + 0.08 * math.sin(elapsed * 4.2)
    colour = tuple(portal.get("colour", portal.get("color", (80, 245, 255))))

    glow_radius = int(radius * 3.7 * pulse)
    glow = pygame.Surface((glow_radius * 2 + 4, glow_radius * 2 + 4), pygame.SRCALPHA)
    local_center = (glow.get_width() // 2, glow.get_height() // 2)
    for layer in range(7, 0, -1):
        pygame.draw.circle(
            glow,
            (*colour, 5 + (8 - layer) * 5),
            local_center,
            max(radius, int(glow_radius * layer / 7)),
        )
    screen.blit(glow, (center[0] - local_center[0], center[1] - local_center[1]))

    outer = int(radius * 1.55 * pulse)
    pygame.draw.circle(screen, (25, 98, 135), center, outer, max(2, radius // 7))
    pygame.draw.circle(screen, colour, center, radius, max(3, radius // 6))
    pygame.draw.circle(screen, (3, 9, 27), center, max(2, int(radius * 0.67)))
    pygame.draw.circle(screen, (174, 254, 255), center, max(2, int(radius * 0.39)), 2)

    rect = pygame.Rect(center[0] - outer, center[1] - outer, outer * 2, outer * 2)
    for index in range(3):
        start = elapsed * (1.1 + index * 0.18) + index * math.tau / 3
        pygame.draw.arc(screen, (155, 250, 255), rect, start, start + 0.82, 3)


def draw_ship(
    screen: Any,
    position: tuple[float, float],
    angle: float,
    thrust: tuple[float, float],
    ship_radius: float,
) -> None:
    assert pygame is not None
    x, y = position
    radius = max(9.0, ship_radius * 1.35)
    forward = (math.cos(angle), math.sin(angle))
    side = (-forward[1], forward[0])

    def point(longitudinal: float, lateral: float) -> tuple[int, int]:
        return (
            round(x + forward[0] * longitudinal + side[0] * lateral),
            round(y + forward[1] * longitudinal + side[1] * lateral),
        )

    glow_radius = round(radius * 2.5)
    glow = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(glow, (68, 207, 255, 28), (glow_radius, glow_radius), glow_radius)
    screen.blit(glow, (round(x) - glow_radius, round(y) - glow_radius))

    thrust_magnitude = math.hypot(*thrust)
    if thrust_magnitude > 0.0:
        thrust_direction = (thrust[0] / thrust_magnitude, thrust[1] / thrust_magnitude)
        exhaust_side = (-thrust_direction[1], thrust_direction[0])
        flame_length = radius * (1.35 + 0.20 * math.sin(pygame.time.get_ticks() * 0.04))
        base_x = x - thrust_direction[0] * radius * 0.55
        base_y = y - thrust_direction[1] * radius * 0.55
        flame = [
            (
                round(base_x + exhaust_side[0] * radius * 0.25),
                round(base_y + exhaust_side[1] * radius * 0.25),
            ),
            (
                round(x - thrust_direction[0] * flame_length),
                round(y - thrust_direction[1] * flame_length),
            ),
            (
                round(base_x - exhaust_side[0] * radius * 0.25),
                round(base_y - exhaust_side[1] * radius * 0.25),
            ),
        ]
        pygame.draw.polygon(screen, (255, 97, 50), flame)

    hull = [
        point(radius, 0),
        point(-radius * 0.68, -radius * 0.72),
        point(-radius * 0.34, 0),
        point(-radius * 0.68, radius * 0.72),
    ]
    pygame.draw.polygon(screen, WHITE, hull)
    pygame.draw.lines(screen, (62, 191, 255), True, hull, 2)
    pygame.draw.circle(screen, (144, 239, 255), point(radius * 0.28, 0), max(2, int(radius * 0.18)))


def keyboard_action() -> tuple[float, float]:
    assert pygame is not None
    keys = pygame.key.get_pressed()
    thrust_x = float(keys[pygame.K_d] or keys[pygame.K_RIGHT]) - float(
        keys[pygame.K_a] or keys[pygame.K_LEFT]
    )
    thrust_y = float(keys[pygame.K_s] or keys[pygame.K_DOWN]) - float(
        keys[pygame.K_w] or keys[pygame.K_UP]
    )
    magnitude = math.hypot(thrust_x, thrust_y)
    if magnitude > 1.0:
        thrust_x /= magnitude
        thrust_y /= magnitude
    return thrust_x, thrust_y


def distance_to_portal(env: GravityEnv) -> float:
    ship_x, ship_y = _xy(env.ship_position)
    portal_x, portal_y = _position(env.portal)
    return math.hypot(portal_x - ship_x, portal_y - ship_y)


def status_text(env: GravityEnv) -> tuple[str, tuple[int, int, int]]:
    status = str(env.status).replace("_", " ").upper()
    if env.success:
        return "PORTAL REACHED", GREEN
    if env.done:
        if "TIME" in status:
            return "TIME LIMIT", (255, 201, 89)
        return status or "SHIP LOST", RED
    return "IN FLIGHT", CYAN


def translucent_panel(screen: Any, rect: Any) -> None:
    assert pygame is not None
    panel = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(panel, (5, 12, 31, 216), panel.get_rect(), border_radius=12)
    pygame.draw.rect(panel, (55, 117, 169, 120), panel.get_rect(), 1, border_radius=12)
    screen.blit(panel, rect.topleft)


def draw_hud(
    screen: Any,
    env: GravityEnv,
    seed: int,
    fonts: tuple[Any, Any],
    attempts: list[AttemptTrail],
    *,
    generation: int | None,
    policy_version: int | None,
    source_label: str,
) -> None:
    assert pygame is not None
    title_font, text_font = fonts
    vx, vy = _xy(env.ship_velocity)
    speed = math.hypot(vx, vy)
    status, status_colour = status_text(env)
    completed_rewards = [
        float(attempt.reward) for attempt in attempts if attempt.reward is not None
    ]
    average_reward = (
        f"{sum(completed_rewards) / len(completed_rewards):8.2f}" if completed_rewards else "     N/A"
    )
    best_reward = f"{max(completed_rewards):8.2f}" if completed_rewards else "     N/A"
    known_outcomes = [attempt.success for attempt in attempts if attempt.success is not None]
    success_rate = (
        f"{100.0 * sum(bool(outcome) for outcome in known_outcomes) / len(known_outcomes):6.1f}%"
        if known_outcomes
        else "   N/A"
    )
    info = env.info()
    clearance_seen = float(info["min_clearance_seen"])
    generation_text = str(generation) if generation is not None else "N/A"
    policy_text = f"v{policy_version}" if policy_version is not None else "MANUAL"

    panel = pygame.Rect(18, 18, 445, 286)
    translucent_panel(screen, panel)
    screen.blit(title_font.render("GRAVITY GAUNTLET", True, WHITE), (34, 29))

    lines = (
        (f"SOURCE       {source_label}", MUTED),
        (f"SEED / STEP  {seed} / {env.timestep}", WHITE),
        (f"GEN / POLICY {generation_text} / {policy_text}", WHITE),
        (f"CURRENT RWD  {env.episode_reward:8.2f}", WHITE),
        (f"ALL AVG/BEST {average_reward} / {best_reward}", WHITE),
        (f"ALL SUCCESS  {success_rate}  ({len(known_outcomes)} outcomes)", WHITE),
        (f"VELOCITY     ({vx:6.1f}, {vy:6.1f})", WHITE),
        (f"SPEED        {speed:8.2f}", WHITE),
        (f"TARGET       {distance_to_portal(env):8.1f} px", WHITE),
        (f"MIN CLEAR    {clearance_seen:8.2f} px", WHITE if clearance_seen >= 58 else RED),
        (f"STATUS       {status}", status_colour),
    )
    for index, (text, colour) in enumerate(lines):
        screen.blit(text_font.render(text, True, colour), (35, 61 + index * 20))

    controls = pygame.Rect(18, HEIGHT - 61, 510, 43)
    translucent_panel(screen, controls)
    hint = "WASD / ARROWS  THRUST     R  RESTART     N  NEW UNIVERSE"
    screen.blit(text_font.render(hint, True, MUTED), (34, HEIGHT - 49))


def draw_end_overlay(screen: Any, env: GravityEnv, fonts: tuple[Any, Any], elapsed: float) -> None:
    assert pygame is not None
    if not env.done:
        return

    title_font, text_font = fonts
    center = tuple(round(value) for value in env.ship_position)
    effects = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    if env.success:
        effects.fill((22, 235, 171, 14))
        for index in range(5):
            radius = int(23 + index * 20 + (elapsed * 35) % 20)
            pygame.draw.circle(effects, (80, 255, 201, max(10, 90 - index * 15)), center, radius, 3)
        headline = "PORTAL REACHED"
        detail = "Success — press R to replay or N for a new universe"
        colour = GREEN
    else:
        effects.fill((255, 18, 55, 20))
        pulse = int(30 + 6 * math.sin(elapsed * 7))
        pygame.draw.circle(effects, (255, 57, 82, 110), center, pulse, 4)
        for index in range(12):
            angle = index * math.tau / 12
            inner = (center[0] + round(math.cos(angle) * 18), center[1] + round(math.sin(angle) * 18))
            outer = (center[0] + round(math.cos(angle) * 52), center[1] + round(math.sin(angle) * 52))
            pygame.draw.line(effects, (255, 108, 82, 140), inner, outer, 3)
        headline = "SHIP LOST"
        if "collision" in str(env.status):
            reason = "Collision"
        elif env.status == "out_of_bounds":
            reason = "Lost beyond the flight boundary"
        elif env.status == "timeout":
            reason = "Flight time exhausted"
        else:
            reason = str(env.status).replace("_", " ").title()
        detail = f"{reason} — press R to retry or N for a new universe"
        colour = RED

    screen.blit(effects, (0, 0))
    banner = pygame.Rect(WIDTH // 2 - 285, HEIGHT // 2 - 62, 570, 124)
    translucent_panel(screen, banner)
    title = title_font.render(headline, True, colour)
    subtitle = text_font.render(detail, True, WHITE)
    screen.blit(title, title.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 19)))
    screen.blit(subtitle, subtitle.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 24)))


def reset_world(env: GravityEnv, seed: int) -> None:
    env.reset(seed=seed)


def _completed_attempt(
    env: GravityEnv,
) -> AttemptTrail | None:
    points = tuple((float(point["x"]), float(point["y"])) for point in env.trajectory)
    if len(points) < 2:
        return None
    return AttemptTrail(
        points=points,
        reward=float(env.episode_reward),
        success=bool(env.success),
        seed=int(env.seed),
    )


def _source_label(attempts: list[AttemptTrail], current_seed: int) -> str:
    matching = [attempt for attempt in attempts if attempt.seed == current_seed]
    if any(attempt.sandbox_id is not None for attempt in matching):
        return "LOCAL MANUAL + DAYTONA GHOSTS"
    if matching:
        return "LOCAL MANUAL + REAL GHOSTS"
    return "LOCAL MANUAL FLIGHT"


def run_demo(
    seed: int = 7,
    max_frames: int | None = None,
    *,
    rollout_trails: list[AttemptTrail] | None = None,
    generation: int | None = None,
    policy_version: int | None = None,
) -> None:
    """Run the game; ``max_frames`` supports a bounded headless smoke test."""

    if pygame is None:
        raise SystemExit("Pygame is required. Install the project dependencies, then run python visual_demo.py")

    pygame.init()
    try:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Gravity Gauntlet — 2D Space MVP")
        clock = pygame.time.Clock()
        fonts = (
            pygame.font.SysFont("menlo", 24, bold=True),
            pygame.font.SysFont("menlo", 15),
        )

        current_seed = int(seed)
        env = GravityEnv(seed=current_seed)
        reset_world(env, current_seed)
        background, stars = make_background(current_seed)
        trail: deque[tuple[float, float]] = deque([_xy(env.ship_position)], maxlen=TRAIL_LENGTH)
        attempts = list(rollout_trails or [])
        ship_angle = -math.pi / 2
        frame_count = 0
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        completed = _completed_attempt(env)
                        if completed is not None:
                            attempts.append(completed)
                            attempts = attempts[-24:]
                        reset_world(env, current_seed)
                        trail = deque([_xy(env.ship_position)], maxlen=TRAIL_LENGTH)
                    elif event.key == pygame.K_n:
                        completed = _completed_attempt(env)
                        if completed is not None:
                            attempts.append(completed)
                            attempts = attempts[-24:]
                        current_seed = (current_seed + 1) % 2_147_483_647
                        reset_world(env, current_seed)
                        background, stars = make_background(current_seed)
                        trail = deque([_xy(env.ship_position)], maxlen=TRAIL_LENGTH)

            thrust = keyboard_action()
            if not env.done:
                # GravityEnv is the single source of truth for every state change.
                env.step(thrust)

            position = _xy(env.ship_position)
            velocity = _xy(env.ship_velocity)
            if math.hypot(*velocity) > 0.05:
                ship_angle = math.atan2(velocity[1], velocity[0])
            if math.hypot(position[0] - trail[-1][0], position[1] - trail[-1][1]) >= 0.25:
                trail.append(position)

            elapsed = pygame.time.get_ticks() / 1000.0
            screen.blit(background, (0, 0))
            draw_stars(screen, stars, elapsed)
            draw_ghost_trails(screen, attempts, current_seed)
            draw_trail(screen, trail)
            draw_planets(screen, env.planets, current_seed)
            draw_asteroids(screen, env.asteroids, current_seed)
            draw_portal(screen, env.portal, elapsed)
            draw_speed_streaks(screen, position, velocity)
            draw_ship(screen, position, ship_angle, thrust, float(env.ship_radius))
            draw_champion_label(screen, attempts, current_seed, fonts[1])
            draw_hud(
                screen,
                env,
                current_seed,
                fonts,
                attempts,
                generation=generation,
                policy_version=policy_version,
                source_label=_source_label(attempts, current_seed),
            )
            draw_end_overlay(screen, env, fonts, elapsed)
            pygame.display.flip()

            clock.tick(FPS)
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                running = False
    finally:
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Play the deterministic Gravity Gauntlet visual MVP.")
    parser.add_argument(
        "--seed",
        type=int,
        help="universe seed (default: loaded champion's seed, otherwise 7)",
    )
    parser.add_argument(
        "--rollouts",
        type=Path,
        help="optional real worker/generation JSON whose trajectories become ghost trails",
    )
    parser.add_argument("--generation", type=int, help="real generation number to display")
    parser.add_argument("--policy-version", type=int, help="real policy version to display")
    args = parser.parse_args()
    rollout_trails = load_rollout_trails(args.rollouts) if args.rollouts else []
    seed = args.seed
    if seed is None:
        scored_with_seed = [
            attempt
            for attempt in rollout_trails
            if attempt.seed is not None and attempt.reward is not None
        ]
        if scored_with_seed:
            seed = int(max(scored_with_seed, key=lambda attempt: float(attempt.reward)).seed)
        else:
            known_seeds = [attempt.seed for attempt in rollout_trails if attempt.seed is not None]
            seed = int(known_seeds[0]) if known_seeds else 7

    policy_version = args.policy_version
    if policy_version is None:
        known_versions = {
            attempt.policy_version
            for attempt in rollout_trails
            if attempt.policy_version is not None
        }
        if len(known_versions) == 1:
            policy_version = known_versions.pop()
    generation = args.generation
    if generation is None:
        known_generations = {
            attempt.generation
            for attempt in rollout_trails
            if attempt.generation is not None
        }
        if len(known_generations) == 1:
            generation = known_generations.pop()
    smoke_frames = os.environ.get("GRAVITY_DEMO_MAX_FRAMES")
    run_demo(
        seed,
        int(smoke_frames) if smoke_frames else None,
        rollout_trails=rollout_trails,
        generation=generation,
        policy_version=policy_version,
    )


if __name__ == "__main__":
    main()
