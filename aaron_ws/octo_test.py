import robosuite as suite
import cv2
import jax
import numpy as np
from collections import deque
from octo.model.octo_model import OctoModel

print("Loading Octo model...")

# model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")

print("Model loaded!")

language_instruction = "Pick up the L-shape tool, and use it to pull the red cube closer"

WINDOW_SIZE = 2

task = model.create_tasks(texts=[language_instruction])

env = suite.make(
    "CustomHookPull",
    robots="Panda",
    has_renderer=False,  # Turn off on-screen rendering to avoid segfault
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=["agentview", "robot0_eye_in_hand"],
    camera_heights=256,
    camera_widths=256,
)

obs = env.reset()
agentview_history = deque(maxlen=WINDOW_SIZE)
wrist_history = deque(maxlen=WINDOW_SIZE)

for _ in range(WINDOW_SIZE):
    agentview_history.append(obs["agentview_image"])
    wrist_history.append(obs["robot0_eye_in_hand_image"])


print(f"Environment action_dim: {env.action_dim}")

for episode in range(5):
    print(f"\n=== Episode {episode} ===")
    obs = env.reset()

    # Reset histories
    agentview_history.clear()
    wrist_history.clear()

    for _ in range(WINDOW_SIZE):
        agentview_history.append(obs["agentview_image"])
        # Resize wrist camera from 256x256 to 128x128
        wrist_resized = cv2.resize(obs["robot0_eye_in_hand_image"], (128, 128))
        wrist_history.append(wrist_resized)

    done = False
    step_count = 0

    while not done and step_count < 500:

        agentview_images = np.stack(list(agentview_history))[None]  # [1, 2, 256, 256, 3]
        wrist_images = np.stack(list(wrist_history))[None]  # [1, 2, 256, 256, 3]

        observation = {
            'image_primary': agentview_images,
            'image_wrist': wrist_images,
            'timestep_pad_mask': np.full((1, WINDOW_SIZE), True, dtype=bool),
            'pad_mask_dict': {
                'image_primary': np.full((1, WINDOW_SIZE), True, dtype=bool),
                'image_wrist': np.full((1, WINDOW_SIZE), True, dtype=bool),
            }
        }

        actions = model.sample_actions(
            observation, 
            task,
            rng=jax.random.PRNGKey(episode * 1000 + step_count)
        )

        # Extract first action from action chunk
        action = np.array(actions[0, 0])  # [batch=0, chunk=0, :]
        # Clip actions to reasonable range (Octo sometimes outputs large values)

        action = np.clip(action, -1.0, 1.0)
        obs, reward, done, info = env.step(action)
        
        # Show interactive viewer
        # env.render()
        
        # Update both histories (resize wrist)
        agentview_history.append(obs["agentview_image"])
        wrist_resized = cv2.resize(obs["robot0_eye_in_hand_image"], (128, 128))
        wrist_history.append(wrist_resized)

        # Visualize every 10 steps

        if step_count % 10 == 0:
            agentview = cv2.cvtColor(obs["agentview_image"], cv2.COLOR_RGB2BGR)
            wrist = cv2.cvtColor(obs["robot0_eye_in_hand_image"], cv2.COLOR_RGB2BGR)
            combined = np.hstack([agentview, wrist])

            cv2.putText(combined, f"Ep {episode} Step {step_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(combined, f"Reward: {reward:.3f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Octo Policy', combined)
            key = cv2.waitKey(1)

            if key & 0xFF == ord('q'):
                print("User quit")
                break

        step_count += 1


    success = info.get('success', False)
    print(f"Episode {episode} finished in {step_count} steps. Success: {success}")

cv2.destroyAllWindows()
env.close()
print("Done!")