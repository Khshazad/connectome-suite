"""
Embodied Bio-Robotics Closed-Loop Simulation Runner
===================================================

Executes multi-step closed-loop 2D and 3D agent navigation simulations using
Central Complex EPG Ring Attractor, lobula plate EMD, CPG motor generator, and Giant Fiber escape circuits.
Telemetry and trajectory logs are saved to `simulation_results.json`.
"""

import json
import logging
from pathlib import Path
import importlib.util

_sim_dir = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("simulator", _sim_dir / "simulator.py")
_sim_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sim_mod)

Environment2D = _sim_mod.Environment2D
Environment3D = _sim_mod.Environment3D
EmbodiedAgent = _sim_mod.EmbodiedAgent

logger = logging.getLogger("BioRoboticsSimulation")


def run_robotics_simulation(num_steps: int = 200, output_dir: Path = None) -> dict:
    """
    Executes closed-loop 2D and 3D agent navigation simulations.

    Args:
        num_steps (int): Number of timesteps to simulate. Defaults to 200.
        output_dir (Path, optional): Directory to output simulation_results.json.

    Returns:
        dict: Complete simulation summary and trajectory logs for 2D and 3D runs.
    """
    logger.info(f"Starting closed-loop 2D continuous simulation ({num_steps} steps)...")
    env_2d = Environment2D(width=100.0, height=100.0)
    agent_2d = EmbodiedAgent(x=10.0, y=10.0, heading=0.0)

    escape_events_2d = 0
    epg_nav_steps_2d = 0

    for step in range(num_steps):
        log = agent_2d.step_2d(env_2d, dt=0.5)
        if log["control_mode"] == "GIANT_FIBER_ESCAPE":
            escape_events_2d += 1
        else:
            epg_nav_steps_2d += 1

        dist_to_goal = ((agent_2d.x - env_2d.goal["x"])**2 + (agent_2d.y - env_2d.goal["y"])**2)**0.5
        if dist_to_goal < env_2d.goal["radius"]:
            logger.info(f"✓ 2D Agent reached target goal at step {step}!")
            break

    dist_to_goal_2d = ((agent_2d.x - env_2d.goal["x"])**2 + (agent_2d.y - env_2d.goal["y"])**2)**0.5

    summary_2d = {
        "total_steps_executed": len(agent_2d.trajectory_2d),
        "initial_position": [10.0, 10.0],
        "final_position": [round(agent_2d.x, 2), round(agent_2d.y, 2)],
        "goal_position": [env_2d.goal["x"], env_2d.goal["y"]],
        "final_distance_to_goal": round(dist_to_goal_2d, 2),
        "goal_reached": dist_to_goal_2d < env_2d.goal["radius"],
        "giant_fiber_escapes_triggered": escape_events_2d,
        "epg_navigation_steps": epg_nav_steps_2d,
        "trajectory_sample": agent_2d.trajectory_2d[::10],
    }

    logger.info(f"Starting closed-loop 3D continuous flight simulation ({num_steps} steps)...")
    env_3d = Environment3D(width=100.0, height=100.0, depth=100.0)
    agent_3d = EmbodiedAgent(x=10.0, y=10.0, z=10.0, heading=0.0)

    escape_events_3d = 0
    epg_nav_steps_3d = 0

    for step in range(num_steps):
        log_3d = agent_3d.step_3d(env_3d, dt=0.5)
        if "ESCAPE" in log_3d["control_mode"]:
            escape_events_3d += 1
        else:
            epg_nav_steps_3d += 1

        dist_to_goal_3d = ((agent_3d.x - env_3d.goal["x"])**2 + (agent_3d.y - env_3d.goal["y"])**2 + (agent_3d.z - env_3d.goal["z"])**2)**0.5
        if dist_to_goal_3d < env_3d.goal["radius"]:
            logger.info(f"✓ 3D Flight Agent reached target goal at step {step}!")
            break

    dist_to_goal_3d = ((agent_3d.x - env_3d.goal["x"])**2 + (agent_3d.y - env_3d.goal["y"])**2 + (agent_3d.z - env_3d.goal["z"])**2)**0.5

    summary_3d = {
        "total_steps_executed": len(agent_3d.trajectory_3d),
        "initial_position": [10.0, 10.0, 10.0],
        "final_position": [round(agent_3d.x, 2), round(agent_3d.y, 2), round(agent_3d.z, 2)],
        "goal_position": [env_3d.goal["x"], env_3d.goal["y"], env_3d.goal["z"]],
        "final_distance_to_goal": round(dist_to_goal_3d, 2),
        "goal_reached": dist_to_goal_3d < env_3d.goal["radius"],
        "giant_fiber_escapes_triggered": escape_events_3d,
        "epg_navigation_steps": epg_nav_steps_3d,
        "trajectory_sample": agent_3d.trajectory_3d[::10],
    }

    full_results = {
        "simulation_mode": "2D_and_3D_closed_loop",
        "2d_navigation_summary": summary_2d,
        "3d_flight_summary": summary_3d,
        "neural_telemetry_2d_sample": agent_2d.neural_logs[::5],
        "neural_telemetry_3d_sample": agent_3d.neural_logs[::5],
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "simulation_results.json"
        with open(results_file, "w") as f:
            json.dump(full_results, f, indent=2)
        logger.info(f"Saved closed-loop simulation telemetry to {results_file}")

    return full_results


if __name__ == "__main__":
    out_path = Path("/mnt/data/Connectome/connectome_products/7_embodied_robotics")
    run_robotics_simulation(num_steps=200, output_dir=out_path)
