"""Deterministic 2D physics environment for the Gravity Gauntlet MVP."""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Sequence


class GravityEnv:
    """A small pygame-independent environment with a Gymnasium-style API.

    ``reset()`` returns ``(observation, info)`` and ``step(action)`` returns
    ``(observation, reward, terminated, truncated, info)``.  Layout generation
    is seeded and the physics contains no random term, so a seed plus an action
    sequence always produces the same trajectory.
    """

    WIDTH = 1200
    HEIGHT = 800

    SHIP_RADIUS = 10.0
    THRUST_ACCELERATION = 105.0
    BASE_GRAVITY_PARAMETER = 480_000.0
    GRAVITY_SOFTENING = 35.0

    PLANET_COLOURS = (
        (255, 105, 120),
        (255, 176, 75),
        (255, 225, 90),
        (104, 220, 255),
        (148, 118, 255),
        (102, 240, 178),
    )
    ASTEROID_COLOURS = (
        (151, 145, 160),
        (173, 158, 145),
        (126, 137, 153),
    )

    def __init__(
        self,
        seed: int = 0,
        max_steps: int = 7_200,
        dt: float = 1.0 / 60.0,
    ) -> None:
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be a positive finite number")

        self.width = self.WIDTH
        self.height = self.HEIGHT
        self.ship_radius = self.SHIP_RADIUS
        self.seed = int(seed)

        self.ship_position: list[float]
        self.ship_velocity: list[float]
        self.planets: list[dict[str, Any]]
        self.asteroids: list[dict[str, Any]]
        self.portal: dict[str, Any]
        self.timestep: int
        self.done: bool
        self.success: bool
        self.status: str
        self.collision: dict[str, Any] | None
        self._terminated: bool
        self._truncated: bool
        self._last_action: list[float]
        self._last_gravity: list[float]
        self._last_thrust: list[float]

        self.reset()

    @property
    def state(self) -> dict[str, Any]:
        """Return an isolated, JSON-safe state snapshot."""

        return self._observation()

    @property
    def current_timestep(self) -> int:
        return self.timestep

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Rebuild a universe and return ``(observation, info)``.

        With no seed, the current universe is reconstructed exactly.  Passing
        a seed selects a new deterministic universe.  ``options`` is accepted
        for compatibility with common RL environment callers.
        """

        del options
        if seed is not None:
            self.seed = int(seed)

        self._generate_universe(random.Random(self.seed))
        self.timestep = 0
        self.done = False
        self.success = False
        self.status = "running"
        self.collision = None
        self._terminated = False
        self._truncated = False
        self._last_action = [0.0, 0.0]
        self._last_gravity = [0.0, 0.0]
        self._last_thrust = [0.0, 0.0]
        return self._observation(), self._info()

    def step(
        self, action: Sequence[float]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Advance one deterministic physics step.

        Action components are independently clamped to ``[-1, 1]``.  The
        integration order is exactly:

        ``velocity += (gravity + thrust) * dt``
        ``position += velocity * dt``
        """

        applied_action = self._clamp_action(action)
        if self.done:
            self._last_action = applied_action
            return (
                self._observation(),
                0.0,
                self._terminated,
                self._truncated,
                self._info(),
            )

        previous_position = list(self.ship_position)
        gravity_x, gravity_y = self.gravity_at(previous_position)
        thrust_x = applied_action[0] * self.THRUST_ACCELERATION
        thrust_y = applied_action[1] * self.THRUST_ACCELERATION

        self.ship_velocity[0] += (gravity_x + thrust_x) * self.dt
        self.ship_velocity[1] += (gravity_y + thrust_y) * self.dt
        self.ship_position[0] += self.ship_velocity[0] * self.dt
        self.ship_position[1] += self.ship_velocity[1] * self.dt
        self.timestep += 1

        self._last_action = applied_action
        self._last_gravity = [gravity_x, gravity_y]
        self._last_thrust = [thrust_x, thrust_y]
        self._detect_terminal_state(previous_position)

        return (
            self._observation(),
            self._event_reward(),
            self._terminated,
            self._truncated,
            self._info(),
        )

    def gravity_at(self, position: Sequence[float]) -> tuple[float, float]:
        """Return total softened inverse-square gravity at ``position``.

        Every planet contributes the required acceleration:

        ``a = GM * r / (|r|^2 + epsilon^2)^(3/2)``
        """

        try:
            if len(position) != 2:
                raise ValueError
            x, y = float(position[0]), float(position[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("position must contain exactly two numbers") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("position components must be finite")

        acceleration_x = 0.0
        acceleration_y = 0.0
        epsilon_squared = self.GRAVITY_SOFTENING**2
        for planet in self.planets:
            dx = planet["position"][0] - x
            dy = planet["position"][1] - y
            distance_squared = dx * dx + dy * dy
            denominator = (distance_squared + epsilon_squared) ** 1.5
            scale = planet["gm"] / denominator
            acceleration_x += scale * dx
            acceleration_y += scale * dy
        return acceleration_x, acceleration_y

    def distance_to_target(self) -> float:
        return math.dist(self.ship_position, self.portal["position"])

    def _generate_universe(self, rng: random.Random) -> None:
        self.ship_position = [
            rng.uniform(90.0, 145.0),
            rng.uniform(150.0, self.height - 150.0),
        ]
        self.ship_velocity = [rng.uniform(82.0, 102.0), rng.uniform(-18.0, 18.0)]

        self.portal = {
            "position": [
                rng.uniform(self.width - 145.0, self.width - 75.0),
                rng.uniform(110.0, self.height - 110.0),
            ],
            "radius": rng.uniform(28.0, 34.0),
            "colour": [80, 245, 255],
        }

        occupied: list[tuple[list[float], float]] = [
            (self.ship_position, self.ship_radius + 75.0),
            (self.portal["position"], self.portal["radius"] + 80.0),
        ]

        self.planets = []
        for index in range(rng.randint(3, 4)):
            radius = rng.uniform(36.0, 59.0)
            position = self._sample_clear_position(
                rng,
                (280.0, self.width - 260.0),
                (90.0, self.height - 90.0),
                radius,
                occupied,
                55.0,
            )
            mass = rng.uniform(0.78, 1.28) * (radius / 45.0) ** 2
            planet = {
                "position": position,
                "radius": radius,
                "mass": mass,
                "gm": self.BASE_GRAVITY_PARAMETER * mass,
                "gravity_radius": max(145.0, radius * rng.uniform(3.1, 3.8)),
                "colour": list(rng.choice(self.PLANET_COLOURS)),
                "index": index,
            }
            self.planets.append(planet)
            occupied.append((position, radius))

        self.asteroids = []
        for index in range(rng.randint(8, 11)):
            radius = rng.uniform(12.0, 22.0)
            position = self._sample_clear_position(
                rng,
                (205.0, self.width - 115.0),
                (55.0, self.height - 55.0),
                radius,
                occupied,
                18.0,
            )
            asteroid = {
                "position": position,
                "radius": radius,
                "angle": rng.uniform(0.0, math.tau),
                "colour": list(rng.choice(self.ASTEROID_COLOURS)),
                "index": index,
            }
            self.asteroids.append(asteroid)
            occupied.append((position, radius))

        # Semantic aliases share the same authoritative objects.
        self.gravity_wells = self.planets
        self.obstacles = self.asteroids
        self.target_portal = self.portal

    @staticmethod
    def _sample_clear_position(
        rng: random.Random,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        radius: float,
        occupied: list[tuple[list[float], float]],
        padding: float,
    ) -> list[float]:
        best_position: list[float] | None = None
        best_clearance = -math.inf
        for _ in range(500):
            candidate = [rng.uniform(*x_range), rng.uniform(*y_range)]
            clearance = min(
                math.dist(candidate, other_position) - radius - other_radius
                for other_position, other_radius in occupied
            )
            if clearance >= padding:
                return candidate
            if clearance > best_clearance:
                best_position = candidate
                best_clearance = clearance
        assert best_position is not None
        return best_position

    @staticmethod
    def _clamp_action(action: Sequence[float]) -> list[float]:
        try:
            if len(action) != 2:
                raise ValueError
            values = [float(action[0]), float(action[1])]
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("action must contain exactly two numbers") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError("action components must be finite")
        return [max(-1.0, min(1.0, value)) for value in values]

    def _detect_terminal_state(self, previous_position: Sequence[float]) -> None:
        for index, planet in enumerate(self.planets):
            if self._segment_hits_circle(
                previous_position,
                self.ship_position,
                planet["position"],
                self.ship_radius + planet["radius"],
            ):
                self._finish_collision("planet", index, planet)
                return

        for index, asteroid in enumerate(self.asteroids):
            if self._segment_hits_circle(
                previous_position,
                self.ship_position,
                asteroid["position"],
                self.ship_radius + asteroid["radius"],
            ):
                self._finish_collision("asteroid", index, asteroid)
                return

        if self._segment_hits_circle(
            previous_position,
            self.ship_position,
            self.portal["position"],
            self.ship_radius + self.portal["radius"],
        ):
            self.done = True
            self.success = True
            self.status = "success"
            self._terminated = True
            return

        x, y = self.ship_position
        if x < 0.0 or x > self.width or y < 0.0 or y > self.height:
            self.done = True
            self.status = "out_of_bounds"
            self._terminated = True
            return

        if self.timestep >= self.max_steps:
            self.done = True
            self.status = "timeout"
            self._truncated = True

    @staticmethod
    def _segment_hits_circle(
        start: Sequence[float],
        end: Sequence[float],
        centre: Sequence[float],
        radius: float,
    ) -> bool:
        segment_x = end[0] - start[0]
        segment_y = end[1] - start[1]
        length_squared = segment_x * segment_x + segment_y * segment_y
        if length_squared == 0.0:
            closest_x, closest_y = start[0], start[1]
        else:
            projection = (
                (centre[0] - start[0]) * segment_x
                + (centre[1] - start[1]) * segment_y
            ) / length_squared
            projection = max(0.0, min(1.0, projection))
            closest_x = start[0] + projection * segment_x
            closest_y = start[1] + projection * segment_y
        dx = closest_x - centre[0]
        dy = closest_y - centre[1]
        return dx * dx + dy * dy <= radius * radius

    def _finish_collision(
        self,
        kind: str,
        index: int,
        body: dict[str, Any],
    ) -> None:
        self.collision = {
            "kind": kind,
            "index": index,
            "position": list(body["position"]),
        }
        self.done = True
        self.status = f"collision_{kind}"
        self._terminated = True

    def _event_reward(self) -> float:
        """Minimal event signal only; no learning or reward shaping lives here."""

        if self.success:
            return 1.0
        if self.collision is not None or self.status == "out_of_bounds":
            return -1.0
        if self._truncated:
            return -0.1
        return -0.001

    def _observation(self) -> dict[str, Any]:
        return {
            "world_size": [self.width, self.height],
            "seed": self.seed,
            "ship_position": list(self.ship_position),
            "ship_velocity": list(self.ship_velocity),
            "ship_radius": self.ship_radius,
            "planets": copy.deepcopy(self.planets),
            "asteroids": copy.deepcopy(self.asteroids),
            "portal": copy.deepcopy(self.portal),
            "timestep": self.timestep,
            "done": self.done,
            "success": self.success,
            "status": self.status,
        }

    def _info(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "timestep": self.timestep,
            "status": self.status,
            "success": self.success,
            "collision": copy.deepcopy(self.collision),
            "distance_to_target": self.distance_to_target(),
            "action": list(self._last_action),
            "gravity_acceleration": list(self._last_gravity),
            "thrust_acceleration": list(self._last_thrust),
        }


__all__ = ["GravityEnv"]
