import robosuite as suite
import numpy as np
import cv2

env = suite.make(
    "CustomHookPull",
    robots="Panda",
    has_renderer=True,  # Interactive viewer
    has_offscreen_renderer=True,  # Camera images
    use_camera_obs=True,
    camera_names=["agentview", "robot0_eye_in_hand", "frontview", "birdview"],
    camera_heights=256,
    camera_widths=256,
)

obs = env.reset()

print("Controls:")
print("  Rotate the on-screen viewer with your mouse")
print("  Compare it to the fixed camera views in the other window")
print("  Press 'q' in the camera window to quit")

for i in range(1000):
    action = np.random.randn(env.action_dim) * 0.1
    obs, reward, done, info = env.step(action)
    
    # Show interactive viewer
    env.render()
    
    # Show fixed cameras
    agentview = cv2.cvtColor(obs["agentview_image"], cv2.COLOR_RGB2BGR)
    wrist = cv2.cvtColor(obs["robot0_eye_in_hand_image"], cv2.COLOR_RGB2BGR)
    frontview = cv2.cvtColor(obs["frontview_image"], cv2.COLOR_RGB2BGR)
    birdview = cv2.cvtColor(obs["birdview_image"], cv2.COLOR_RGB2BGR)
    
    # Arrange in 2x2 grid
    top_row = np.hstack([agentview, wrist])
    bottom_row = np.hstack([frontview, birdview])
    combined = np.vstack([top_row, bottom_row])
    
    # Add labels
    cv2.putText(combined, "AgentView", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(combined, "Wrist", (270, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(combined, "FrontView", (10, 286), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(combined, "BirdView", (270, 286), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow('Fixed Camera Views (Compare to On-Screen Viewer)', combined)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if done:
        obs = env.reset()

cv2.destroyAllWindows()
env.close()