"""
Embodied Bio-Robotics Simulator
================================

Closed-loop 2D embodied agent simulator controlled by a fruit fly bio-inspired spiking
neural controller (Central Complex EPG Ring Attractor, EMD Visual Correlator, Giant Fiber Escape).
"""

import math
import torch
import numpy as np

import sys
import importlib.util
from pathlib import Path

# Load Product 2 circuit library modules
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


class Environment2D:
    """Bounded 2D continuous environment with obstacles and target goal."""

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
        """Check if position collides with walls or obstacles."""
        if x - radius < 0 or x + radius > self.width or y - radius < 0 or y + radius > self.height:
            return True
        for obs in self.obstacles:
            dist = math.hypot(x - obs["x"], y - obs["y"])
            if dist < (radius + obs["radius"]):
                return True
        return False

    def cast_rays(self, x: float, y: float, heading: float, num_rays: int = 16, max_dist: float = 40.0):
        """Cast multi-directional distance sensing rays around agent."""
        distances = []
        ray_angles = [heading + (2 * math.pi * i / num_rays) for i in range(num_rays)]
        
        for angle in ray_angles:
            min_d = max_dist
            dx = math.cos(angle)
            dy = math.sin(angle)
            
            # Check step distances
            for step in range(1, int(max_dist)):
                rx = x + dx * step
                ry = y + dy * step
                if self.check_collision(rx, ry, radius=0.5):
                    min_d = float(step)
                    break
            distances.append(min_d)
            
        return torch.tensor(distances, dtype=torch.float32)


class EmbodiedAgent:
    """Embodied bio-robotic agent driven by Central Complex & escape circuits."""

    def __init__(self, x: float = 10.0, y: float = 10.0, heading: float = 0.0):
        self.x = x
        self.y = y
        self.heading = heading  # radians
        self.radius = 2.0
        self.speed = 0.0

        # Bio-inspired Spiking Neural Controller
        self.ring_attractor = EPGRingAttractor(num_wedges=16)
        self.ring_attractor.reset_state(initial_heading_rad=heading)

        self.emd = HassensteinReichardtEMD(num_channels=16)
        self.emd.reset_state()

        self.giant_fiber = GiantFiberEscapeCircuit()
        self.giant_fiber.reset_state()

        self.trajectory = []
        self.neural_logs = []

    def step(self, env: Environment2D, dt: float = 1.0):
        """Single closed-loop simulation step."""
        # 1. Sensory Perception (Raycasting & Target Angle)
        ray_dists = env.cast_rays(self.x, self.y, self.heading, num_rays=16, max_dist=40.0)
        
        # Front obstacle proximity and looming threat calculation
        front_dist = min(ray_dists[0].item(), ray_dists[1].item(), ray_dists[-1].item())
        looming_expansion = max(0.0, (40.0 - front_dist) / 10.0)

        # Goal target vector
        dx_goal = env.goal["x"] - self.x
        dy_goal = env.goal["y"] - self.y
        goal_angle = math.atan2(dy_goal, dx_goal) % (2 * math.pi)

        # 2. Run Central Complex EPG Ring Attractor
        # Inject external goal cue into EPG ring attractor
        goal_wedge = int((goal_angle / (2 * math.pi)) * 16) % 16
        cue_tensor = torch.zeros(16)
        cue_tensor[goal_wedge] = 8.0

        spikes_epg, v_epg, decoded_heading_rad, decoded_heading_deg = self.ring_attractor(
            angular_velocity=0.0, external_cue=cue_tensor
        )

        # 3. Run EMD Visual Motion Correlator
        # Inverse distance luminance input across 16 channels
        luminance = 1.0 / (ray_dists + 1e-3)
        emd_net, spikes_left, spikes_right, net_motion = self.emd(luminance)

        # 4. Run Giant Fiber Escape Circuit
        gf_spike, ttm_jump, dlm_wings, gf_threat = self.giant_fiber(angular_expansion_rate=looming_expansion)

        # 5. Motor Control & Kinematics Integration
        if gf_spike.item() > 0 or ttm_jump.item() > 0:
            # Emergency Giant Fiber Escape maneuver: sharp 90-degree evasive turn & speed burst
            steering_omega = 1.5  # rapid turn rate
            linear_v = 4.0        # jump burst speed
            control_mode = "GIANT_FIBER_ESCAPE"
        else:
            # Heading error between EPG ring bump and current heading
            heading_error = math.atan2(math.sin(goal_angle - self.heading), math.cos(goal_angle - self.heading))
            
            # EMD obstacle avoidance steering bias
            avoidance_steering = -0.05 * net_motion

            steering_omega = 0.8 * heading_error + avoidance_steering
            linear_v = 1.5 if front_dist > 10.0 else 0.5
            control_mode = "EPG_EMD_NAV"

        # Update physical kinematic state
        self.heading = (self.heading + steering_omega * dt) % (2 * math.pi)
        new_x = self.x + linear_v * math.cos(self.heading) * dt
        new_y = self.y + linear_v * math.sin(self.heading) * dt

        # Collision update
        if not env.check_collision(new_x, new_y, self.radius):
            self.x = new_x
            self.y = new_y

        self.speed = linear_v
        self.trajectory.append((self.x, self.y, self.heading))

        log_entry = {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "heading_deg": round(math.degrees(self.heading), 2),
            "epg_decoded_heading_deg": round(decoded_heading_deg, 2),
            "front_dist": round(front_dist, 2),
            "net_motion": round(net_motion, 4),
            "gf_threat": round(gf_threat, 2),
            "control_mode": control_mode,
        }
        self.neural_logs.append(log_entry)
        return log_entry
