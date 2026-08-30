"""Headless regression tests for deterministic Gravity Gauntlet physics."""

from __future__ import annotations

import json
import math
import unittest

from gravity_env import (
    ACTION_VECTORS,
    CURRICULUM_LEVELS,
    MAX_ASTEROIDS,
    MAX_PLANETS,
    OBSERVATION_DIM,
    GravityEnv,
    curriculum_level_for_seed,
)


def _planet(
    position: list[float],
    *,
    radius: float = 30.0,
    mass: float = 1.0,
    gm: float = GravityEnv.BASE_GRAVITY_PARAMETER,
) -> dict[str, object]:
    return {
        "position": position,
        "radius": radius,
        "mass": mass,
        "gm": gm,
        "gravity_radius": 200.0,
        "colour": [255, 255, 255],
        "index": 0,
    }


class GravityFieldTests(unittest.TestCase):
    def test_gravity_points_toward_planet_and_matches_softened_formula(self) -> None:
        env = GravityEnv(seed=0, max_steps=20)
        planet_position = [600.0, 400.0]
        env.planets = [_planet(planet_position)]
        env.asteroids = []
        sample = [800.0, 450.0]

        acceleration = env.gravity_at(sample)
        displacement = (
            planet_position[0] - sample[0],
            planet_position[1] - sample[1],
        )
        distance_squared = displacement[0] ** 2 + displacement[1] ** 2
        scale = env.BASE_GRAVITY_PARAMETER / (
            distance_squared + env.GRAVITY_SOFTENING**2
        ) ** 1.5

        self.assertGreater(
            acceleration[0] * displacement[0]
            + acceleration[1] * displacement[1],
            0.0,
        )
        self.assertAlmostEqual(
            acceleration[0] * displacement[1]
            - acceleration[1] * displacement[0],
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            acceleration[0], scale * displacement[0], places=12
        )
        self.assertAlmostEqual(
            acceleration[1], scale * displacement[1], places=12
        )

        near = math.hypot(*env.gravity_at([700.0, 400.0]))
        far = math.hypot(*env.gravity_at([1_000.0, 400.0]))
        self.assertGreater(near, far)

    def test_real_gravity_curves_a_coasting_trajectory(self) -> None:
        curved = GravityEnv(seed=0, max_steps=200)
        straight = GravityEnv(seed=0, max_steps=200)
        for env, gm in (
            (curved, curved.BASE_GRAVITY_PARAMETER),
            (straight, 0.0),
        ):
            env.ship_position = [300.0, 300.0]
            env.ship_velocity = [100.0, 0.0]
            env.planets = [_planet([600.0, 400.0], radius=20.0, gm=gm)]
            env.asteroids = []
            env.portal = {"position": [1_100.0, 700.0], "radius": 25.0}

        for _ in range(90):
            curved.step((0.0, 0.0))
            straight.step((0.0, 0.0))

        self.assertGreater(curved.ship_position[1], straight.ship_position[1] + 1.0)
        self.assertGreater(math.dist(curved.ship_position, straight.ship_position), 4.0)
        self.assertEqual(curved.info()["thrust_acceleration"], [0.0, 0.0])

    def test_empty_gravity_coast_is_exactly_ballistic(self) -> None:
        env = GravityEnv(seed=0, max_steps=100)
        env.ship_position = [300.0, 500.0]
        env.ship_velocity = [100.0, -20.0]
        env.planets = []
        env.asteroids = []
        env.portal = {"position": [1_100.0, 700.0], "radius": 25.0}

        start_position = list(env.ship_position)
        start_velocity = list(env.ship_velocity)
        step_count = 30
        for _ in range(step_count):
            env.step((0.0, 0.0))

        self.assertAlmostEqual(
            env.ship_position[0],
            start_position[0] + start_velocity[0] * env.dt * step_count,
        )
        self.assertAlmostEqual(
            env.ship_position[1],
            start_position[1] + start_velocity[1] * env.dt * step_count,
        )
        self.assertEqual(env.ship_velocity, start_velocity)
        self.assertEqual(env.info()["gravity_acceleration"], [0.0, 0.0])
        self.assertEqual(env.info()["thrust_acceleration"], [0.0, 0.0])

    def test_production_strength_medium_body_creates_local_flyby(self) -> None:
        curved = GravityEnv(seed=0, max_steps=300)
        ballistic = GravityEnv(seed=0, max_steps=300)
        body_mass = 0.9
        for env, gm in (
            (curved, curved.BASE_GRAVITY_PARAMETER * body_mass),
            (ballistic, 0.0),
        ):
            env.ship_position = [300.0, 310.0]
            env.ship_velocity = [130.0, 0.0]
            env.planets = [
                _planet(
                    [600.0, 400.0],
                    radius=36.0,
                    mass=body_mass,
                    gm=gm,
                )
            ]
            env.asteroids = []
            env.portal = {"position": [1_100.0, 700.0], "radius": 25.0}

        for _ in range(240):
            curved.step((0.0, 0.0))
            ballistic.step((0.0, 0.0))

        self.assertEqual(curved.status, "running")
        self.assertGreater(
            math.dist(curved.ship_position, ballistic.ship_position),
            40.0,
        )
        self.assertGreater(curved.max_speed, 140.0)
        self.assertGreater(curved.min_clearance_seen, 0.0)
        self.assertEqual(curved.info()["thrust_acceleration"], [0.0, 0.0])


class ProceduralUniverseTests(unittest.TestCase):
    def test_same_seed_and_actions_reproduce_universe_and_trajectory(self) -> None:
        first = GravityEnv(seed=18_473, max_steps=200)
        second = GravityEnv(seed=18_473, max_steps=200)
        different = GravityEnv(seed=18_474, max_steps=200)

        self.assertEqual(first.universe_dict(), second.universe_dict())
        self.assertNotEqual(first.universe_dict(), different.universe_dict())

        for action in (3, 3, 2, 0, 4, 5, 0, 1):
            first.step_discrete(action)
            second.step_discrete(action)
            if first.done or second.done:
                break
        self.assertEqual(first.trajectory, second.trajectory)
        self.assertEqual(first.info(), second.info())

        original = first.universe_dict()
        first.reset(seed=18_473)
        self.assertEqual(first.universe_dict(), original)

    def test_seed_bands_form_a_reconstructable_curriculum(self) -> None:
        worlds = [GravityEnv(seed=level) for level in range(CURRICULUM_LEVELS)]

        for level, env in enumerate(worlds):
            self.assertEqual(curriculum_level_for_seed(env.seed), level)
            self.assertEqual(env.curriculum_level, level)
            self.assertEqual(env.universe_dict()["curriculum_level"], level)
            planet_range = env.PLANET_COUNT_RANGES[level]
            asteroid_range = env.ASTEROID_COUNT_RANGES[level]
            portal_range = env.PORTAL_RADIUS_RANGES[level]
            self.assertIn(
                len(env.planets),
                range(planet_range[0], planet_range[1] + 1),
            )
            self.assertIn(
                len(env.asteroids),
                range(asteroid_range[0], asteroid_range[1] + 1),
            )
            self.assertGreaterEqual(env.portal["radius"], portal_range[0])
            self.assertLessEqual(env.portal["radius"], portal_range[1])

        self.assertLess(len(worlds[0].planets), len(worlds[-1].planets))
        self.assertEqual(len(worlds[0].asteroids), 1)
        self.assertGreaterEqual(len(worlds[-1].asteroids), 2)
        self.assertGreater(worlds[0].portal["radius"], worlds[-1].portal["radius"])

    def test_distributed_bodies_are_bounded_and_radius_correlated(self) -> None:
        self.assertGreaterEqual(GravityEnv.BASE_GRAVITY_PARAMETER, 300_000.0)
        self.assertLessEqual(GravityEnv.BASE_GRAVITY_PARAMETER, 400_000.0)
        for seed in range(128):
            env = GravityEnv(seed=seed)
            self.assertGreaterEqual(len(env.planets), 4)
            self.assertLessEqual(len(env.planets), MAX_PLANETS)
            self.assertGreaterEqual(len(env.asteroids), 1)
            self.assertLessEqual(len(env.asteroids), MAX_ASTEROIDS)

            masses = [float(planet["mass"]) for planet in env.planets]
            self.assertLess(max(masses) / sum(masses), 0.39)
            self.assertGreaterEqual(
                max(float(planet["position"][0]) for planet in env.planets)
                - min(float(planet["position"][0]) for planet in env.planets),
                env.width * 0.35,
            )

            for planet in env.planets:
                self.assertNotIn("hero", planet)
                radius = float(planet["radius"])
                mass = float(planet["mass"])
                radius_range = env.PLANET_RADIUS_RANGES[env.curriculum_level]
                self.assertGreaterEqual(radius, radius_range[0])
                self.assertLessEqual(radius, radius_range[1])
                self.assertGreaterEqual(mass, env.MIN_PLANET_MASS)
                self.assertLessEqual(mass, env.MAX_PLANET_MASS)
                self.assertAlmostEqual(
                    float(planet["gm"]),
                    env.BASE_GRAVITY_PARAMETER * mass,
                )

                radius_fraction = (
                    radius - env.MIN_PLANET_RADIUS
                ) / (
                    env.MAX_PLANET_RADIUS - env.MIN_PLANET_RADIUS
                )
                radius_mass = env.MIN_RADIUS_MASS + radius_fraction * (
                    env.MAX_RADIUS_MASS - env.MIN_RADIUS_MASS
                )
                expected_low = min(
                    env.MAX_PLANET_MASS,
                    max(
                        env.MIN_PLANET_MASS,
                        radius_mass * env.PLANET_MASS_VARIATION[0],
                    ),
                )
                expected_high = min(
                    env.MAX_PLANET_MASS,
                    max(
                        env.MIN_PLANET_MASS,
                        radius_mass * env.PLANET_MASS_VARIATION[1],
                    ),
                )
                self.assertGreaterEqual(mass, expected_low - 1.0e-12)
                self.assertLessEqual(mass, expected_high + 1.0e-12)
                self.assertLessEqual(float(planet["gravity_radius"]), 126.0)

            for asteroid in env.asteroids:
                self.assertGreaterEqual(float(asteroid["radius"]), 10.0)
                self.assertLessEqual(float(asteroid["radius"]), 18.0)

    def test_no_body_controls_most_of_the_far_arena(self) -> None:
        winner_coverages = []
        maximum_contribution_shares = []
        for seed in range(64):
            env = GravityEnv(seed=seed)
            winner_counts = [0] * len(env.planets)
            contribution_shares = []
            far_sample_count = 0
            for x in range(100, env.width, 70):
                for y in range(80, env.height, 64):
                    point = [float(x), float(y)]
                    if any(
                        math.dist(point, planet["position"])
                        - float(planet["radius"])
                        < 120.0
                        for planet in env.planets
                    ):
                        continue

                    contributions = []
                    for planet in env.planets:
                        dx = float(planet["position"][0]) - point[0]
                        dy = float(planet["position"][1]) - point[1]
                        distance_squared = dx * dx + dy * dy
                        contributions.append(
                            float(planet["gm"])
                            * math.sqrt(distance_squared)
                            / (
                                distance_squared + env.GRAVITY_SOFTENING**2
                            )
                            ** 1.5
                        )
                    winner = max(range(len(contributions)), key=contributions.__getitem__)
                    winner_counts[winner] += 1
                    contribution_shares.append(
                        max(contributions) / sum(contributions)
                    )
                    far_sample_count += 1

            winner_coverages.append(max(winner_counts) / far_sample_count)
            maximum_contribution_shares.append(
                sum(contribution_shares) / len(contribution_shares)
            )

        self.assertLess(sum(winner_coverages) / len(winner_coverages), 0.40)
        self.assertLess(max(winner_coverages), 0.60)
        self.assertLess(
            sum(maximum_contribution_shares)
            / len(maximum_contribution_shares),
            0.55,
        )
        self.assertLess(max(maximum_contribution_shares), 0.65)


class ObservationAndRewardTests(unittest.TestCase):
    def test_observation_fields_and_zero_padding_have_fixed_meaning(self) -> None:
        env = GravityEnv(seed=0)
        env.ship_position = [100.0, 100.0]
        env.ship_velocity = [50.0, -25.0]
        env.portal = {"position": [1_100.0, 700.0], "radius": 30.0}
        env.planets = [_planet([300.0, 200.0], radius=40.0, mass=2.0)]
        env.asteroids = [
            {
                "position": [100.0, 300.0],
                "radius": 20.0,
                "angle": 0.0,
                "colour": [150, 150, 150],
                "index": 0,
            }
        ]

        observation = env.get_observation()
        diagonal = math.hypot(env.width, env.height)

        self.assertEqual(len(observation), OBSERVATION_DIM)
        expected_navigation = [
            100.0 / env.width,
            100.0 / env.height,
            50.0 / env.VELOCITY_SCALE,
            -25.0 / env.VELOCITY_SCALE,
            1_000.0 / env.width,
            600.0 / env.height,
            math.hypot(1_000.0, 600.0) / diagonal,
        ]
        for actual, expected in zip(observation[:7], expected_navigation):
            self.assertAlmostEqual(actual, expected)

        planet_clearance = math.hypot(200.0, 100.0) - 40.0 - env.ship_radius
        expected_planet = [
            200.0 / env.width,
            100.0 / env.height,
            planet_clearance / diagonal,
            2.0 / env.MASS_SCALE,
            40.0 / env.RADIUS_SCALE,
        ]
        for actual, expected in zip(observation[7:12], expected_planet):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(
            observation[12 : 7 + MAX_PLANETS * 5],
            [0.0] * ((MAX_PLANETS - 1) * 5),
        )

        asteroid_start = 7 + MAX_PLANETS * 5
        expected_asteroid = [
            0.0,
            200.0 / env.height,
            (200.0 - 20.0 - env.ship_radius) / diagonal,
            20.0 / env.RADIUS_SCALE,
        ]
        for actual, expected in zip(
            observation[asteroid_start : asteroid_start + 4], expected_asteroid
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(
            observation[asteroid_start + 4 :],
            [0.0] * ((MAX_ASTEROIDS - 1) * 4),
        )

    def test_progress_reward_is_normalized_by_map_diagonal(self) -> None:
        env = GravityEnv(seed=0)
        env.planets = []
        env.asteroids = []
        previous_distance = env.distance_to_target() + 10.0

        reward = env._shaped_reward(previous_distance, (0.0, 0.0))
        expected = (
            env.STEP_COST
            + (10.0 / math.hypot(env.width, env.height)) * env.PROGRESS_SCALE
        )
        self.assertAlmostEqual(reward, expected)

    def test_representative_episodes_remain_finite_and_strict_json(self) -> None:
        for seed in range(8):
            env = GravityEnv(seed=seed, max_steps=160)
            for decision in range(40):
                observation, reward, terminated, truncated, info = env.step_discrete(
                    decision % len(ACTION_VECTORS)
                )
                self.assertTrue(all(math.isfinite(value) for value in observation))
                self.assertTrue(math.isfinite(reward))
                self.assertTrue(math.isfinite(info["episode_reward"]))
                if terminated or truncated:
                    break
            json.dumps(
                {
                    "universe": env.universe_dict(),
                    "trajectory": env.trajectory,
                    "info": env.info(),
                },
                allow_nan=False,
            )


class SweptTerminalEventTests(unittest.TestCase):
    @staticmethod
    def _fast_environment() -> GravityEnv:
        env = GravityEnv(seed=0, max_steps=10, dt=1.0)
        env.ship_position = [100.0, 100.0]
        env.ship_velocity = [400.0, 0.0]
        env.asteroids = []
        return env

    def test_fast_segment_collision_cannot_tunnel_through_planet(self) -> None:
        env = self._fast_environment()
        env.planets = [_planet([300.0, 100.0], radius=30.0, gm=0.0)]
        env.portal = {"position": [1_100.0, 700.0], "radius": 20.0}

        _, reward, terminated, truncated, _ = env.step((0.0, 0.0))

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(env.status, "collision_planet")
        self.assertLess(reward, -100.0)

    def test_fast_segment_portal_crossing_succeeds(self) -> None:
        env = self._fast_environment()
        env.planets = []
        env.portal = {"position": [300.0, 100.0], "radius": 30.0}

        _, reward, terminated, truncated, _ = env.step((0.0, 0.0))

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(env.success)
        self.assertEqual(env.status, "portal")
        self.assertGreater(reward, env.PORTAL_BONUS)


if __name__ == "__main__":
    unittest.main()
