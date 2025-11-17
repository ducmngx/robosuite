import robosuite as suite
import numpy as np

env = suite.make(
    env_name= "Lift",#"CustomHookPull",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
    horizon=1000,
)

# Reset the environment to place objects
obs = env.reset()

# Run the simulation loop
for i in range(10000):
    # Random action
    action = np.random.randn(env.action_dim) * 0.1
    
    # Step the environment
    obs, reward, done, info = env.step(action)
    
    # Render
    env.render()
    
    if done:
        obs = env.reset()

env.close()