"""
Embodied Bio-Robotics Closed-Loop Simulation Runner
===================================================

Executes full multi-step closed-loop simulation of embodied fly agent navigating
in 2D arena with obstacles using Central Complex EPG, EMD, and Giant Fiber circuits.
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
EmbodiedAgent = _sim_mod.EmbodiedAgent

logger = logging.getLogger("BioRoboticsSimulation")


def run_robotics_simulation(num_steps: int = 200, output_dir: Path = None):
    """
    Run closed-loop agent navigation simulation.
    
    Args:
        num_steps (int): Number of timesteps to simulate.
        output_dir (Path, optional): Directory to save simulation output logs.
        
    Returns:
        dict: Simulation summary and trajectory logs.
    """
    env = Environment2D(width=100.0, height=100.0)
    agent = EmbodiedAgent(x=10.0, y=10.0, heading=0.0)

    logger.info(f"Starting closed-loop embodied simulation for {num_steps} steps...")
    
    escape_events = 0
    epg_nav_steps = 0
    collisions = 0

    for step in range(num_steps):
        log = agent.step(env, dt=0.5)
        
        if log["control_mode"] == "GIANT_FIBER_ESCAPE":
            escape_events += 1
        else:
            epg_nav_steps += 1

        dist_to_goal = ((agent.x - env.goal["x"])**2 + (agent.y - env.goal["y"])**2)**0.5
        if dist_to_goal < env.goal["radius"]:
            logger.info(f"✓ Agent reached target goal at step {step}!")
            break

    dist_to_goal = ((agent.x - env.goal["x"])**2 + (agent.y - env.goal["y"])**2)**0.5

    summary = {
        "total_steps_executed": len(agent.trajectory),
        "initial_position": [10.0, 10.0],
        "final_position": [round(agent.x, 2), round(agent.y, 2)],
        "goal_position": [env.goal["x"], env.goal["y"]],
        "final_distance_to_goal": round(dist_to_goal, 2),
        "goal_reached": dist_to_goal < env.goal["radius"],
        "giant_fiber_escapes_triggered": escape_events,
        "epg_navigation_steps": epg_nav_steps,
        "trajectory_sample": agent.trajectory[::10],
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "simulation_results.json"
        with open(results_file, "w") as f:
            json.dump({"summary": summary, "neural_logs_sample": agent.neural_logs[::5]}, f, indent=2)
        logger.info(f"Saved simulation results to {results_file}")

    return summary


if __name__ == "__main__":
    out_path = Path("/mnt/data/Connectome/connectome_products/7_embodied_robotics")
    run_robotics_simulation(num_steps=200, output_dir=out_path)
