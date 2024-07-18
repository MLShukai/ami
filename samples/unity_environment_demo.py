import sys
import time

import torch

from ami.interactions.environments.unity_environment import UnityEnvironment


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python unity_environment_sample.py <unity_env_file_path>")
        sys.exit(1)

    unity_env_file_path = sys.argv[1]
    print(f"Unity environment file path: {unity_env_file_path}")

    # Create and initialize the Unity environment
    env = UnityEnvironment(file_path=unity_env_file_path, worker_id=0, time_scale=1.0, target_frame_rate=60)

    env.setup()

    # Display information about the action and observation spaces
    print("Action Space: ", env.gym_unity_environment.action_space)
    print("Observation Space:", env.gym_unity_environment.observation_space)

    # Run for 100 frames
    for i in range(100):
        start_time = time.perf_counter()

        # Generate a random action and convert it to a Tensor
        action = torch.from_numpy(env.gym_unity_environment.action_space.sample())

        # Apply the action to the environment
        env.affect(action)

        # Get the new observation
        observation = env.observe()

        # Calculate and display frame rate and elapsed time
        elapsed = time.perf_counter() - start_time
        fps = 1 / elapsed
        print(f"Frame {i}: {fps:.2f} FPS, Elapsed time: {elapsed:.4f} seconds")
        print(f"Observation shape: {observation.shape}")

    # Clean up the environment
    env.teardown()


if __name__ == "__main__":
    main()
