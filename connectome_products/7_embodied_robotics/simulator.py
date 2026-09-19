"""
Embodied Bio-Robotics Closed-Loop Simulator
============================================

Closed-loop 2D/3D embodied fly agent simulator driven by a fruit fly bio-inspired spiking
neural controller:
1. EPGRingAttractor: Central Complex EPG 360-degree heading direction tracking.
2. HassensteinReichardtEMD: Optic lobe Elementary Motion Detector visual correlator.
3. GiantFiberEscapeCircuit: Giant Fiber rapid looming threat escape motor circuit.
4. CPGMotorGenerator: Central Pattern Generator rhythmic motor propulsion.
"""

import math
import torch
import numpy as np
from pathlib import Path
import sys
import importlib.util
from typing import Dict, List, Tuple, Optional

# Dynamically load Product 2 circuit library package
_p2_dir = Path(__file__).parent.parent / "2_circuit_library"
_spec = importlib.util.spec_from_file_location(
    "connectome_products.2_circuit_library",
    _p2_dir / "__init__.py",
    submodule_search_locations=[str(_p2_dir)]
)
_circuit_lib = importlib.util.module_from_spec(_spec)
sys.modules["connectome_products.2_circuit_library"] = _circuit_lib
_spec.loader.exec_module(_circuit_lib)

EPGRingAttractor = _circuit_lib.EPGRingAttractor
HassensteinReichardtEMD = _circuit_lib.HassensteinReichardtEMD
GiantFiberEscapeCircuit = _circuit_lib.GiantFiberEscapeCircuit
CPGMotorGenerator = _circuit_lib.CPGMotorGenerator


class Environment2D:
    """Bounded continuous 2D spatial environment with circular obstacles and goal."""

    def __init__(self, width: float = 100.0, height: float = 100.0):
        self.width = width
        self.height = height
        self.obstacles = [
            {"x": 30.0, "y": 40.0, "radius": 8.0},
            {"x": 60.0, "y": 70.0, "radius": 10.0},
            {"x": 75.0, "y": 30.0, "radius": 7.0},
            {"x": 45.0, "y": 20.0, "radius": 6.0},
            {"x": 20.0, "y": 80.0, "radius": 9.0},
        ]
        self.goal = {"x": 90.0, "y": 90.0, "radius": 5.0}

    def check_collision(self, x: float, y: float, radius: float = 2.0) -> bool:
        """Check if position collides with boundary walls or obstacles."""
        if x - radius < 0 or x + radius > self.width or y - radius < 0 or y + radius > self.height:
            return True
        for obs in self.obstacles:
            dist = math.hypot(x - obs["x"], y - obs["y"])
            if dist < (radius + obs["radius"]):
                return True
        return False

    def cast_rays(self, x: float, y: float, heading: float, num_rays: int = 16, max_dist: float = 40.0) -> torch.Tensor:
        """Cast multi-directional sensing rays around agent in 2D plane."""
        distances = []
        ray_angles = [heading + (2 * math.pi * i / num_rays) for i in range(num_rays)]

        for angle in ray_angles:
            min_d = max_dist
            dx = math.cos(angle)
            dy = math.sin(angle)

            for step in range(1, int(max_dist)):
                rx = x + dx * step
                ry = y + dy * step
                if self.check_collision(rx, ry, radius=0.5):
                    min_d = float(step)
                    break
            distances.append(min_d)

        return torch.tensor(distances, dtype=torch.float32)


class Environment3D:
    """Continuous 3D spatial flight arena with spherical obstacles and 3D goal."""

    def __init__(self, width: float = 100.0, height: float = 100.0, depth: float = 100.0):
        self.width = width
        self.height = height
        self.depth = depth
        self.obstacles = [
            {"x": 30.0, "y": 40.0, "z": 50.0, "radius": 8.0},
            {"x": 60.0, "y": 70.0, "z": 40.0, "radius": 10.0},
            {"x": 75.0, "y": 30.0, "z": 60.0, "radius": 7.0},
            {"x": 45.0, "y": 20.0, "z": 30.0, "radius": 6.0},
            {"x": 20.0, "y": 80.0, "z": 70.0, "radius": 9.0},
        ]
        self.goal = {"x": 90.0, "y": 90.0, "z": 80.0, "radius": 6.0}

    def check_collision(self, x: float, y: float, z: float, radius: float = 2.0) -> bool:
        """Check if 3D position collides with bounding box or spherical obstacles."""
        if (x - radius < 0 or x + radius > self.width or
            y - radius < 0 or y + radius > self.height or
            z - radius < 0 or z + radius > self.depth):
            return True
        for obs in self.obstacles:
            dist = math.sqrt((x - obs["x"])**2 + (y - obs["y"])**2 + (z - obs["z"])**2)
            if dist < (radius + obs["radius"]):
                return True
        return False

    def cast_rays_3d(self, x: float, y: float, z: float, yaw: float, pitch: float, num_rays: int = 16, max_dist: float = 40.0) -> torch.Tensor:
        """Cast 3D sensory rays distributed across horizontal and vertical angles."""
        distances = []
        angles = [yaw + (2 * math.pi * i / num_rays) for i in range(num_rays)]

        for angle in angles:
            min_d = max_dist
            dx = math.cos(angle) * math.cos(pitch)
            dy = math.sin(angle) * math.cos(pitch)
            dz = math.sin(pitch)

            for step in range(1, int(max_dist)):
                rx = x + dx * step
                ry = y + dy * step
                rz = z + dz * step
                if self.check_collision(rx, ry, rz, radius=0.5):
                    min_d = float(step)
                    break
            distances.append(min_d)

        return torch.tensor(distances, dtype=torch.float32)


class EmbodiedAgent:
    """
    Embodied Bio-Robotic Fly Agent with 2D/3D Kinematics and SNN Controller.

    Integrates EPG Ring Attractor, EMD Visual Motion Detector, Giant Fiber Escape,
    and CPG Rhythmic Propulsion.
    """

    def __init__(self, x: float = 10.0, y: float = 10.0, z: float = 10.0, heading: float = 0.0):
        self.x = x
        self.y = y
        self.z = z
        self.heading = heading  # yaw angle in radians
        self.pitch = 0.0        # pitch angle in radians
        self.roll = 0.0         # roll angle in radians
        self.radius = 2.0
        self.speed = 0.0

        # Bio-inspired Spiking Neural Controller
        self.ring_attractor = EPGRingAttractor(num_wedges=16)
        self.ring_attractor.reset_state(initial_heading_rad=heading)

        self.emd = HassensteinReichardtEMD(num_channels=16)
        self.emd.reset_state()

        self.giant_fiber = GiantFiberEscapeCircuit()
        self.giant_fiber.reset_state()

        self.cpg = CPGMotorGenerator(num_limbs=6)
        self.cpg.reset_state()

        self.trajectory_2d: List[Tuple[float, float, float]] = []
        self.trajectory_3d: List[Tuple[float, float, float, float, float]] = []
        self.neural_logs: List[Dict] = []

    def step_2d(self, env: Environment2D, dt: float = 1.0) -> Dict:
        """
        Executes single step of closed-loop 2D navigation simulation.

        Args:
            env (Environment2D): 2D spatial environment instance.
            dt (float): Timestep delta.

        Returns:
            Dict: Step telemetry log containing agent state and neural activity.
        """
        # 1. Sensory Perception (Raycasting & Goal Heading)
        ray_dists = env.cast_rays(self.x, self.y, self.heading, num_rays=16, max_dist=40.0)
        front_dist = min(ray_dists[0].item(), ray_dists[1].item(), ray_dists[-1].item())
        looming_expansion = max(0.0, (40.0 - front_dist) / 10.0)

        # Target Goal Angle
        dx_goal = env.goal["x"] - self.x
        dy_goal = env.goal["y"] - self.y
        goal_angle = math.atan2(dy_goal, dx_goal) % (2 * math.pi)

        # 2. Central Complex Ring Attractor Update
        goal_wedge = int((goal_angle / (2 * math.pi)) * 16) % 16
        cue_tensor = torch.zeros(16)
        cue_tensor[goal_wedge] = 8.0

        spikes_epg, v_epg, decoded_rad, decoded_deg = self.ring_attractor(
            angular_velocity=0.0, external_cue=cue_tensor
        )

        # 3. Optic Lobe EMD Motion Correlator Update
        luminance = 1.0 / (ray_dists + 1e-3)
        emd_net, spikes_left, spikes_right, net_motion = self.emd(luminance)

        # 4. Giant Fiber Escape Circuit Update
        gf_spike, ttm_jump, dlm_wings, gf_threat = self.giant_fiber(angular_expansion_rate=looming_expansion)

        # 5. CPG Rhythmic Motor Propulsion Update
        f_spikes, e_spikes, cpg_metrics = self.cpg(tonic_drive=18.0)

        # 6. Motor Control Integration
        if gf_spike.item() > 0 or ttm_jump.item() > 0:
            steering_omega = 1.5   # Rapid evasive turn
            linear_v = 4.0         # Escape burst speed
            control_mode = "GIANT_FIBER_ESCAPE"
        else:
            heading_error = math.atan2(math.sin(goal_angle - self.heading), math.cos(goal_angle - self.heading))
            avoidance_steering = -0.05 * net_motion
            steering_omega = 0.8 * heading_error + avoidance_steering
            cpg_propulsion = cpg_metrics["flexor_activity"] * 2.0
            linear_v = (1.5 if front_dist > 10.0 else 0.5) + cpg_propulsion
            control_mode = "EPG_EMD_NAV"

        # Kinematics Update
        self.heading = (self.heading + steering_omega * dt) % (2 * math.pi)
        new_x = self.x + linear_v * math.cos(self.heading) * dt
        new_y = self.y + linear_v * math.sin(self.heading) * dt

        if not env.check_collision(new_x, new_y, self.radius):
            self.x = new_x
            self.y = new_y

        self.speed = linear_v
        self.trajectory_2d.append((round(self.x, 2), round(self.y, 2), round(self.heading, 2)))

        log_entry = {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "heading_deg": round(math.degrees(self.heading), 2),
            "epg_decoded_heading_deg": round(decoded_deg, 2),
            "front_dist": round(front_dist, 2),
            "net_motion": round(net_motion, 4),
            "gf_threat": round(gf_threat, 2),
            "cpg_flexor_activity": cpg_metrics["flexor_activity"],
            "control_mode": control_mode,
        }
        self.neural_logs.append(log_entry)
        return log_entry

    def step_3d(self, env: Environment3D, dt: float = 1.0) -> Dict:
        """
        Executes single step of closed-loop 3D flight navigation simulation.

        Args:
            env (Environment3D): 3D spatial flight arena instance.
            dt (float): Timestep delta.

        Returns:
            Dict: Step telemetry log containing 3D agent state.
        """
        ray_dists = env.cast_rays_3d(self.x, self.y, self.z, self.heading, self.pitch, num_rays=16, max_dist=40.0)
        front_dist = min(ray_dists[0].item(), ray_dists[1].item(), ray_dists[-1].item())
        looming_expansion = max(0.0, (40.0 - front_dist) / 10.0)

        # 3D Vector to Goal
        dx = env.goal["x"] - self.x
        dy = env.goal["y"] - self.y
        dz = env.goal["z"] - self.z
        dist_xy = math.hypot(dx, dy)
        goal_yaw = math.atan2(dy, dx) % (2 * math.pi)
        goal_pitch = math.atan2(dz, dist_xy)

        # Run EPG Ring Attractor for 3D Yaw Direction
        goal_wedge = int((goal_yaw / (2 * math.pi)) * 16) % 16
        cue_tensor = torch.zeros(16)
        cue_tensor[goal_wedge] = 8.0
        _, _, _, decoded_deg = self.ring_attractor(angular_velocity=0.0, external_cue=cue_tensor)

        # Run Giant Fiber for 3D Escape
        gf_spike, ttm_jump, dlm_wings, gf_threat = self.giant_fiber(angular_expansion_rate=looming_expansion)

        if gf_spike.item() > 0:
            yaw_omega = 1.5
            pitch_omega = 0.5
            linear_v = 4.5
            control_mode = "GIANT_FIBER_ESCAPE_3D"
        else:
            yaw_error = math.atan2(math.sin(goal_yaw - self.heading), math.cos(goal_yaw - self.heading))
            pitch_error = goal_pitch - self.pitch
            yaw_omega = 0.8 * yaw_error
            pitch_omega = 0.5 * pitch_error
            linear_v = 2.0 if front_dist > 10.0 else 0.8
            control_mode = "EPG_3D_NAV"

        self.heading = (self.heading + yaw_omega * dt) % (2 * math.pi)
        self.pitch = max(-math.pi / 4, min(math.pi / 4, self.pitch + pitch_omega * dt))

        new_x = self.x + linear_v * math.cos(self.heading) * math.cos(self.pitch) * dt
        new_y = self.y + linear_v * math.sin(self.heading) * math.cos(self.pitch) * dt
        new_z = self.z + linear_v * math.sin(self.pitch) * dt

        if not env.check_collision(new_x, new_y, new_z, self.radius):
            self.x = new_x
            self.y = new_y
            self.z = new_z

        self.trajectory_3d.append((round(self.x, 2), round(self.y, 2), round(self.z, 2), round(self.heading, 2), round(self.pitch, 2)))

        log_entry = {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "z": round(self.z, 2),
            "heading_deg": round(math.degrees(self.heading), 2),
            "pitch_deg": round(math.degrees(self.pitch), 2),
            "front_dist": round(front_dist, 2),
            "gf_threat": round(gf_threat, 2),
            "control_mode": control_mode,
        }
        self.neural_logs.append(log_entry)
        return log_entry
