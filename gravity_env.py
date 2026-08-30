"""Deterministic 2D physics environment for the Gravity Gauntlet MVP."""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Sequence


MAX_PLANETS = 5
MAX_ASTEROIDS = 3
ACTION_HOLD_STEPS = 4
_DIAGONAL = 1.0 / math.sqrt(2.0)
ACTION_VECTORS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.0, -1.0),
    (_DIAGONAL, -_DIAGONAL),
    (1.0, 0.0),
    (_DIAGONAL, _DIAGONAL),
    (0.0, 1.0),
    (-_DIAGONAL, _DIAGONAL),
    (-1.0, 0.0),
    (-_DIAGONAL, -_DIAGONAL),
)

# 7 navigation values + five 5-value planet slots + three 4-value obstacles.
OBSERVATION_DIM = 7 + MAX_PLANETS * 5 + MAX_ASTEROIDS * 4


def action_to_vector(action: int) -> tuple[float, float]:
    """Map coast plus eight compass actions to normalized continuous thrust."""

    if isinstance(action, bool) or not isinstance(action, int):
        raise TypeError("discrete action must be an integer from 0 through 8")
    if not 0 <= action < len(ACTION_VECTORS):
        raise ValueError("discrete action must be an integer from 0 through 8")
    return ACTION_VECTORS[action]


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
    THRUST_ACCELERATION = 135.0
    BASE_GRAVITY_PARAMETER = 620_000.0
    GRAVITY_SOFTENING = 32.0

    SAFE_MARGIN = 58.0
    STEP_COST = -0.002
    PROGRESS_SCALE = 0.022
    THRUST_COST = 0.0015
    SAFETY_SCALE = 0.035
    PORTAL_BONUS = 300.0
    COLLISION_PENALTY = -150.0
    OUT_OF_BOUNDS_PENALTY = -90.0
    TIMEOUT_PENALTY = -25.0
    VELOCITY_SCALE = 500.0
    MASS_SCALE = 4.0
    RADIUS_SCALE = 80.0
    ACTION_HOLD_STEPS = ACTION_HOLD_STEPS
    OBSERVATION_DIM = OBSERVATION_DIM

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
        self._last_discrete_action: int
        self.decision_count: int
        self.episode_reward: float
        self.fuel_used: float
        self.min_clearance_seen: float
        self.max_speed: float
        self.speed_sum: float
        self.trajectory: list[dict[str, Any]]

        self.reset()

    @property
    def state(self) -> dict[str, Any]:
        """Return an isolated, JSON-safe state snapshot."""

        return self._observation()

    @property
    def current_timestep(self) -> int:
        return self.timestep

    def info(self) -> dict[str, Any]:
        """Return current diagnostics without advancing the environment."""

        return self._info()

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
        self.initial_ship_position = list(self.ship_position)
        self.initial_ship_velocity = list(self.ship_velocity)
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
        self._last_discrete_action = 0
        self.decision_count = 0
        self.episode_reward = 0.0
        self.fuel_used = 0.0
        self.min_clearance_seen = self.minimum_clearance()
        self.max_speed = math.hypot(*self.ship_velocity)
        self.speed_sum = self.max_speed
        self.trajectory = [self._trajectory_point(0.0)]
        return self._observation(), self._info()

    def step(
        self, action: int | Sequence[float]
    ) -> tuple[dict[str, Any] | list[float], float, bool, bool, dict[str, Any]]:
        """Advance either one manual physics tick or one held policy action.

        A two-value sequence preserves the original continuous manual-control
        API used by :mod:`visual_demo`. An integer selects one of nine policy
        actions and holds it for :data:`ACTION_HOLD_STEPS` physics substeps.
        """

        if isinstance(action, int) and not isinstance(action, bool):
            return self.step_discrete(action)
        return self._step_physics(action)  # type: ignore[arg-type]

    def step_discrete(
        self,
        action: int,
        hold_steps: int = ACTION_HOLD_STEPS,
    ) -> tuple[list[float], float, bool, bool, dict[str, Any]]:
        """Hold one categorical thrust direction across several real ticks."""

        vector = action_to_vector(action)
        if isinstance(hold_steps, bool) or not isinstance(hold_steps, int) or hold_steps <= 0:
            raise ValueError("hold_steps must be a positive integer")
        if self.done:
            info = self._info()
            info.update({"action_index": action, "physics_steps_held": 0})
            return self.get_observation(), 0.0, self._terminated, self._truncated, info

        total_reward = 0.0
        completed_steps = 0
        for _ in range(hold_steps):
            _, reward, _, _, _ = self._step_physics(vector)
            total_reward += reward
            completed_steps += 1
            if self.done:
                break
        self._last_discrete_action = action
        self.decision_count += 1
        info = self._info()
        info.update(
            {
                "action_index": action,
                "physics_steps_held": completed_steps,
                "decision": self.decision_count,
            }
        )
        return self.get_observation(), total_reward, self._terminated, self._truncated, info

    def _step_physics(
        self,
        action: Sequence[float],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Advance one deterministic semi-implicit Euler physics substep."""

        applied_action = self._clamp_action(action)
        if self.done:
            self._last_action = applied_action
            return self._observation(), 0.0, self._terminated, self._truncated, self._info()

        previous_position = list(self.ship_position)
        previous_distance = self.distance_to_target()
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

        reward = self._shaped_reward(previous_distance, applied_action)
        speed = math.hypot(*self.ship_velocity)
        clearance = self.minimum_clearance()
        self.episode_reward += reward
        self.fuel_used += math.hypot(*applied_action) * self.dt
        self.min_clearance_seen = min(self.min_clearance_seen, clearance)
        self.max_speed = max(self.max_speed, speed)
        self.speed_sum += speed
        self.trajectory.append(self._trajectory_point(reward))
        return self._observation(), reward, self._terminated, self._truncated, self._info()

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

    def surface_clearances(
        self,
        position: Sequence[float] | None = None,
    ) -> list[float]:
        """Return ship-surface clearances for every dangerous body."""

        ship_position = self.ship_position if position is None else position
        clearances = [
            math.dist(ship_position, planet["position"])
            - float(planet["radius"])
            - self.ship_radius
            for planet in self.planets
        ]
        clearances.extend(
            math.dist(ship_position, asteroid["position"])
            - float(asteroid["radius"])
            - self.ship_radius
            for asteroid in self.asteroids
        )
        return clearances

    def minimum_clearance(self, position: Sequence[float] | None = None) -> float:
        clearances = self.surface_clearances(position)
        return min(clearances) if clearances else math.inf

    def get_observation(self) -> list[float]:
        """Return the fixed-size normalized numerical state used by policies."""

        ship_x, ship_y = self.ship_position
        portal_x, portal_y = self.portal["position"]
        diagonal = math.hypot(self.width, self.height)
        observation = [
            ship_x / self.width,
            ship_y / self.height,
            self.ship_velocity[0] / self.VELOCITY_SCALE,
            self.ship_velocity[1] / self.VELOCITY_SCALE,
            (portal_x - ship_x) / self.width,
            (portal_y - ship_y) / self.height,
            self.distance_to_target() / diagonal,
        ]

        for index in range(MAX_PLANETS):
            if index < len(self.planets):
                planet = self.planets[index]
                clearance = (
                    math.dist(self.ship_position, planet["position"])
                    - float(planet["radius"])
                    - self.ship_radius
                )
                observation.extend(
                    (
                        (planet["position"][0] - ship_x) / self.width,
                        (planet["position"][1] - ship_y) / self.height,
                        clearance / diagonal,
                        float(planet["mass"]) / self.MASS_SCALE,
                        float(planet["radius"]) / self.RADIUS_SCALE,
                    )
                )
            else:
                observation.extend((0.0, 0.0, 0.0, 0.0, 0.0))

        for index in range(MAX_ASTEROIDS):
            if index < len(self.asteroids):
                asteroid = self.asteroids[index]
                clearance = (
                    math.dist(self.ship_position, asteroid["position"])
                    - float(asteroid["radius"])
                    - self.ship_radius
                )
                observation.extend(
                    (
                        (asteroid["position"][0] - ship_x) / self.width,
                        (asteroid["position"][1] - ship_y) / self.height,
                        clearance / diagonal,
                        float(asteroid["radius"]) / self.RADIUS_SCALE,
                    )
                )
            else:
                observation.extend((0.0, 0.0, 0.0, 0.0))

        if len(observation) != OBSERVATION_DIM:  # Defensive contract guard.
            raise RuntimeError(
                f"observation has {len(observation)} values; expected {OBSERVATION_DIM}"
            )
        return [float(value) for value in observation]

    def universe_dict(self) -> dict[str, Any]:
        """Return a JSON-safe description separate from mutable ship state."""

        return {
            "world_size": [self.width, self.height],
            "seed": self.seed,
            "initial_ship_position": list(self.initial_ship_position),
            "initial_ship_velocity": list(self.initial_ship_velocity),
            "ship_radius": self.ship_radius,
            "planets": copy.deepcopy(self.planets),
            "asteroids": copy.deepcopy(self.asteroids),
            "portal": copy.deepcopy(self.portal),
        }

    def _generate_universe(self, rng: random.Random) -> None:
        self.ship_position = [
            rng.uniform(90.0, 145.0),
            rng.uniform(150.0, self.height - 150.0),
        ]
        self.ship_velocity = [rng.uniform(68.0, 88.0), rng.uniform(-12.0, 12.0)]

        self.portal = {
            "position": [
                rng.uniform(self.width - 145.0, self.width - 75.0),
                rng.uniform(110.0, self.height - 110.0),
            ],
            "radius": rng.uniform(28.0, 34.0),
            "colour": [80, 245, 255],
        }

        occupied: list[tuple[list[float], float]] = [
            (self.ship_position, self.ship_radius),
            (self.portal["position"], self.portal["radius"]),
        ]

        # Every universe receives a strong but safely offset "hero" gravity
        # well near the broad start-to-portal corridor. Its side and exact
        # placement are seeded, so this creates varied slingshot choices rather
        # than a hard-coded racetrack.
        hero_radius = rng.uniform(49.0, 67.0)
        start_x, start_y = self.ship_position
        portal_x, portal_y = self.portal["position"]
        corridor_x = portal_x - start_x
        corridor_y = portal_y - start_y
        corridor_length = max(1.0, math.hypot(corridor_x, corridor_y))
        along = rng.uniform(0.40, 0.64)
        side = rng.choice((-1.0, 1.0))
        offset = rng.uniform(105.0, 168.0) * side
        hero_position = [
            start_x + corridor_x * along - corridor_y / corridor_length * offset,
            start_y + corridor_y * along + corridor_x / corridor_length * offset,
        ]
        hero_position[0] = max(hero_radius + 75.0, min(self.width - hero_radius - 75.0, hero_position[0]))
        hero_position[1] = max(hero_radius + 65.0, min(self.height - hero_radius - 65.0, hero_position[1]))
        if min(
            math.dist(hero_position, position) - hero_radius - radius
            for position, radius in occupied
        ) < 60.0:
            hero_position = self._sample_clear_position(
                rng,
                (315.0, self.width - 300.0),
                (hero_radius + 70.0, self.height - hero_radius - 70.0),
                hero_radius,
                occupied,
                60.0,
            )
        hero_mass = rng.uniform(1.75, 2.55) * (hero_radius / 56.0) ** 2
        self.planets = [
            {
                "position": hero_position,
                "radius": hero_radius,
                "mass": hero_mass,
                "gm": self.BASE_GRAVITY_PARAMETER * hero_mass,
                "gravity_radius": max(245.0, hero_radius * rng.uniform(4.1, 4.8)),
                "colour": list(rng.choice(self.PLANET_COLOURS)),
                "index": 0,
                "hero": True,
            }
        ]
        occupied.append((hero_position, hero_radius))

        planet_count = rng.randint(2, MAX_PLANETS)
        for index in range(1, planet_count):
            radius = rng.uniform(32.0, 58.0)
            position = self._sample_clear_position(
                rng,
                (250.0, self.width - 220.0),
                (radius + 55.0, self.height - radius - 55.0),
                radius,
                occupied,
                self.SAFE_MARGIN,
            )
            mass = rng.uniform(0.68, 1.28) * (radius / 45.0) ** 2
            self.planets.append(
                {
                    "position": position,
                    "radius": radius,
                    "mass": mass,
                    "gm": self.BASE_GRAVITY_PARAMETER * mass,
                    "gravity_radius": max(150.0, radius * rng.uniform(3.2, 4.0)),
                    "colour": list(rng.choice(self.PLANET_COLOURS)),
                    "index": index,
                    "hero": False,
                }
            )
            occupied.append((position, radius))

        # Keep the corridor well visually and dynamically dominant even when a
        # regular planet happens to draw its largest mass parameters.
        strongest_regular_mass = max(
            (float(planet["mass"]) for planet in self.planets[1:]),
            default=0.0,
        )
        minimum_hero_mass = strongest_regular_mass * 1.20
        if float(self.planets[0]["mass"]) < minimum_hero_mass:
            self.planets[0]["mass"] = minimum_hero_mass
            self.planets[0]["gm"] = self.BASE_GRAVITY_PARAMETER * minimum_hero_mass

        self.asteroids = []
        for index in range(rng.randint(0, MAX_ASTEROIDS)):
            radius = rng.uniform(12.0, 22.0)
            position = self._sample_clear_position(
                rng,
                (205.0, self.width - 115.0),
                (55.0, self.height - 55.0),
                radius,
                occupied,
                self.SAFE_MARGIN,
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
        candidate_position = list(self.ship_position)
        events: list[tuple[float, int, str, int | None, dict[str, Any] | None]] = []
        for index, planet in enumerate(self.planets):
            fraction = self._segment_circle_fraction(
                previous_position,
                candidate_position,
                planet["position"],
                self.ship_radius + planet["radius"],
            )
            if fraction is not None:
                events.append((fraction, 0, "planet", index, planet))

        for index, asteroid in enumerate(self.asteroids):
            fraction = self._segment_circle_fraction(
                previous_position,
                candidate_position,
                asteroid["position"],
                self.ship_radius + asteroid["radius"],
            )
            if fraction is not None:
                events.append((fraction, 1, "asteroid", index, asteroid))

        portal_fraction = self._segment_circle_fraction(
            previous_position,
            candidate_position,
            self.portal["position"],
            self.ship_radius + self.portal["radius"],
        )
        if portal_fraction is not None:
            events.append((portal_fraction, 2, "portal", None, None))

        x, y = candidate_position
        if x < 0.0 or x > self.width or y < 0.0 or y > self.height:
            boundary_fraction = self._boundary_exit_fraction(previous_position, candidate_position)
            events.append((boundary_fraction, 3, "out_of_bounds", None, None))

        if events:
            fraction, _, kind, index, body = min(events, key=lambda event: (event[0], event[1]))
            self.ship_position = [
                previous_position[0] + (candidate_position[0] - previous_position[0]) * fraction,
                previous_position[1] + (candidate_position[1] - previous_position[1]) * fraction,
            ]
            if kind in {"planet", "asteroid"}:
                assert index is not None and body is not None
                self._finish_collision(kind, index, body)
                return
            if kind == "portal":
                self.done = True
                self.success = True
                self.status = "portal"
                self._terminated = True
                return
            self.done = True
            self.status = "out_of_bounds"
            self._terminated = True
            return

        if self.timestep >= self.max_steps:
            self.done = True
            self.status = "timeout"
            self._truncated = True

    @staticmethod
    def _segment_circle_fraction(
        start: Sequence[float],
        end: Sequence[float],
        centre: Sequence[float],
        radius: float,
    ) -> float | None:
        segment_x = end[0] - start[0]
        segment_y = end[1] - start[1]
        length_squared = segment_x * segment_x + segment_y * segment_y
        offset_x = start[0] - centre[0]
        offset_y = start[1] - centre[1]
        radius_squared = radius * radius
        if offset_x * offset_x + offset_y * offset_y <= radius_squared:
            return 0.0
        if length_squared == 0.0:
            return None
        b = 2.0 * (offset_x * segment_x + offset_y * segment_y)
        c = offset_x * offset_x + offset_y * offset_y - radius_squared
        discriminant = b * b - 4.0 * length_squared * c
        if discriminant < 0.0:
            return None
        root = math.sqrt(max(0.0, discriminant))
        fractions = (
            (-b - root) / (2.0 * length_squared),
            (-b + root) / (2.0 * length_squared),
        )
        valid = [fraction for fraction in fractions if 0.0 <= fraction <= 1.0]
        return min(valid) if valid else None

    @classmethod
    def _segment_hits_circle(
        cls,
        start: Sequence[float],
        end: Sequence[float],
        centre: Sequence[float],
        radius: float,
    ) -> bool:
        """Backward-compatible boolean form used by earlier environment tests."""

        return cls._segment_circle_fraction(start, end, centre, radius) is not None

    def _boundary_exit_fraction(
        self,
        start: Sequence[float],
        end: Sequence[float],
    ) -> float:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        candidates: list[float] = []
        if end[0] < 0.0 and dx:
            candidates.append((0.0 - start[0]) / dx)
        elif end[0] > self.width and dx:
            candidates.append((self.width - start[0]) / dx)
        if end[1] < 0.0 and dy:
            candidates.append((0.0 - start[1]) / dy)
        elif end[1] > self.height and dy:
            candidates.append((self.height - start[1]) / dy)
        valid = [fraction for fraction in candidates if 0.0 <= fraction <= 1.0]
        return min(valid) if valid else 1.0

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

    def _shaped_reward(
        self,
        previous_distance: float,
        action: Sequence[float],
    ) -> float:
        """Reward portal progress while teaching safety before impact."""

        progress = previous_distance - self.distance_to_target()
        reward = self.STEP_COST + progress * self.PROGRESS_SCALE
        reward -= math.hypot(*action) * self.THRUST_COST

        clearance = self.minimum_clearance()
        if clearance < self.SAFE_MARGIN:
            danger_fraction = min(1.0, max(0.0, (self.SAFE_MARGIN - clearance) / self.SAFE_MARGIN))
            reward -= self.SAFETY_SCALE * danger_fraction * danger_fraction

        if self.success:
            reward += self.PORTAL_BONUS
        elif self.collision is not None:
            reward += self.COLLISION_PENALTY
        elif self.status == "out_of_bounds":
            reward += self.OUT_OF_BOUNDS_PENALTY
        elif self._truncated:
            reward += self.TIMEOUT_PENALTY
        return float(reward)

    def _event_reward(self) -> float:
        """Backward-compatible terminal-only view of the current event."""

        if self.success:
            return self.PORTAL_BONUS
        if self.collision is not None:
            return self.COLLISION_PENALTY
        if self.status == "out_of_bounds":
            return self.OUT_OF_BOUNDS_PENALTY
        if self._truncated:
            return self.TIMEOUT_PENALTY
        return self.STEP_COST

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
            "rl_observation": self.get_observation(),
        }

    def _info(self) -> dict[str, Any]:
        mean_speed = self.speed_sum / (self.timestep + 1)
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
            "min_clearance": self.minimum_clearance(),
            "min_clearance_seen": self.min_clearance_seen,
            "episode_reward": self.episode_reward,
            "fuel_used": self.fuel_used,
            "speed": math.hypot(*self.ship_velocity),
            "mean_speed": mean_speed,
            "max_speed": self.max_speed,
            "decision": self.decision_count,
        }

    def _trajectory_point(self, reward: float) -> dict[str, Any]:
        return {
            "step": self.timestep,
            "x": float(self.ship_position[0]),
            "y": float(self.ship_position[1]),
            "vx": float(self.ship_velocity[0]),
            "vy": float(self.ship_velocity[1]),
            "reward": float(reward),
            "clearance": float(self.minimum_clearance()),
            "status": self.status,
        }


__all__ = [
    "ACTION_HOLD_STEPS",
    "ACTION_VECTORS",
    "MAX_ASTEROIDS",
    "MAX_PLANETS",
    "OBSERVATION_DIM",
    "GravityEnv",
    "action_to_vector",
]
