from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, HookFrame
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat
import robosuite.utils.transform_utils as T
from robosuite.utils.sim_utils import check_contact


class CustomHookPull(ManipulationEnv):
    """
    This class corresponds to the hook pulling task for a single robot arm.
    
    Task: The robot must use a hook tool to pull a cube that is out of reach
    into its reachable workspace.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        base_types (None or str or list of str): type of base, used to instantiate base models from base factory.
            Default is "default", which is the default base associated with the robot(s) the 'robots' specification.
            None results in no base, and any other (valid) model overrides the default base. Should either be
            single str if same base type is to be used for all robots or else it should be a list of the same
            length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (hook and cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set to False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer
        
        # Store initial cube position for reward calculation
        self.initial_cube_pos = None

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:
            - a discrete reward of 2.25 is provided if the cube is pulled into reach

        Un-normalized summed components if using reward shaping:
            - Reaching hook: in [0, 0.5], to encourage the arm to reach the hook
            - Grasping hook: in {0, 0.25}, non-zero if arm is grasping the hook  
            - Hook to cube: in [0, 0.5], encourages bringing hook close to cube
            - Pulling cube: in [0, 1.0], rewards pulling cube closer to robot

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 2.25

        # use a shaping reward
        elif self.reward_shaping:
            # Stage 1: Reaching hook reward
            hook_pos = self.sim.data.body_xpos[self.hook_body_id]
            gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
            dist_to_hook = np.linalg.norm(gripper_pos - hook_pos)
            reaching_reward = 0.5 * (1 - np.tanh(10.0 * dist_to_hook))
            reward += reaching_reward

            # Stage 2: Grasping hook reward (check if gripper is touching hook)
            # USE CUSTOM CHECK, NOT _check_grasp
            if self._check_hook_grasped():
                reward += 0.25

                # Stage 3: Hook to cube reward (only if hook is grasped)
                cube_pos = self.sim.data.body_xpos[self.cube_body_id]
                # Use the hang site of the hook (the tip where you hook things)
                # hook_hang_site_id = self.sim.model.site_name2id("hook_hang_site")
                hook_hang_site_id = self.sim.model.site_name2id("tool_hang_site")  # was "hook_hang_site"
                hook_hang_pos = self.sim.data.site_xpos[hook_hang_site_id]
                dist_hook_to_cube = np.linalg.norm(hook_hang_pos - cube_pos)
                hook_to_cube_reward = 0.5 * (1 - np.tanh(10.0 * dist_hook_to_cube))
                reward += hook_to_cube_reward

                # Stage 4: Pulling cube reward (reward for bringing cube closer)
                if self.initial_cube_pos is not None:
                    robot_base_pos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    initial_dist = np.linalg.norm(self.initial_cube_pos[:2] - robot_base_pos[:2])
                    current_dist = np.linalg.norm(cube_pos[:2] - robot_base_pos[:2])
                    # Reward proportional to how much closer the cube is now vs initially
                    pulling_progress = (initial_dist - current_dist) / initial_dist
                    pulling_reward = 1.0 * np.clip(pulling_progress, 0, 1)
                    reward += pulling_reward

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

    def _check_hook_grasped(self):
        """
        Check if the robot is grasping the hook.
        Uses contact checking between gripper and hook geoms.
        
        Returns:
            bool: True if hook is grasped
        """
        from robosuite.utils.sim_utils import check_contact
        
        # Get gripper geoms
        gripper_geoms = []
        for arm in self.robots[0].gripper:
            for key in self.robots[0].gripper[arm].important_geoms:
                gripper_geoms.extend(self.robots[0].gripper[arm].important_geoms[key])
        
        # Check contact with hook's grip geom (the part designed to be grasped)
        hook_grip_geom = "tool_grip_frame"  # was "hook_grip_frame"
        return check_contact(self.sim, gripper_geoms, hook_grip_geom)

    # def _load_model(self):
    #     """
    #     Loads an xml model, puts it in self.model
    #     """
    #     super()._load_model()

    #     # Adjust base pose accordingly
    #     xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
    #     self.robots[0].robot_model.set_base_xpos(xpos)

    #     # load model for table top workspace
    #     mujoco_arena = TableArena(
    #         table_full_size=self.table_full_size,
    #         table_friction=self.table_friction,
    #         table_offset=self.table_offset,
    #     )

    #     # Arena always gets set to zero origin
    #     mujoco_arena.set_origin([0, 0, 0])

    #     # initialize hook tool
    #     self.hook = HookFrame(
    #         name="LShapeTool",
    #         frame_length=0.3,
    #         frame_height=0.2,
    #         frame_thickness=0.025,
    #         hook_height=0.05,
    #         grip_location=0.1,
    #         grip_size=(0.02, 0.03),
    #         use_texture=True,
    #     )

    #     # initialize cube (target object to pull)
    #     tex_attrib = {
    #         "type": "cube",
    #     }
    #     mat_attrib = {
    #         "texrepeat": "1 1",
    #         "specular": "0.4",
    #         "shininess": "0.1",
    #     }
    #     redwood = CustomMaterial(
    #         texture="WoodRed",
    #         tex_name="redwood",
    #         mat_name="redwood_mat",
    #         tex_attrib=tex_attrib,
    #         mat_attrib=mat_attrib,
    #     )
    #     self.cube = BoxObject(
    #         name="cube",
    #         size_min=[0.020, 0.020, 0.020],
    #         size_max=[0.022, 0.022, 0.022],
    #         rgba=[1, 0, 0, 1],
    #         material=redwood,
    #         rng=self.rng,
    #     )

    #     # Create placement initializer with sequential sampler
    #     # Hook placed close to robot, cube placed far away (out of reach)
    #     if self.placement_initializer is not None:
    #         self.placement_initializer.reset()
    #         # self.placement_initializer.add_objects([self.hook, self.cube])
    #     else:
    #         self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
    
    #         # Place hook close to robot (within reach)
    #         self.placement_initializer.append_sampler(
    #             UniformRandomSampler(
    #                 name="HookSampler",
    #                 mujoco_objects=self.hook,
    #                 x_range=[-0.10, -0.05],  # Close to robot
    #                 y_range=[-0.15, 0.15],
    #                 rotation=None,
    #                 rotation_axis='z',
    #                 ensure_object_boundary_in_range=False,
    #                 ensure_valid_placement=True,
    #                 reference_pos=self.table_offset,
    #                 z_offset=0.01,
    #                 rng=self.rng,
    #             )
    #         )
            
    #         # Place cube far from robot (out of direct reach, needs hook)
    #         self.placement_initializer.append_sampler(
    #             UniformRandomSampler(
    #                 name="CubeSampler",
    #                 mujoco_objects=self.cube,
    #                 x_range=[0.20, 0.30],  # Far from robot
    #                 y_range=[-0.15, 0.15],
    #                 rotation=None,
    #                 ensure_object_boundary_in_range=False,
    #                 ensure_valid_placement=True,
    #                 reference_pos=self.table_offset,
    #                 z_offset=0.01,
    #                 rng=self.rng,
    #             )
    #         )

    #     # task includes arena, robot, and objects of interest
    #     self.model = ManipulationTask(
    #         mujoco_arena=mujoco_arena,
    #         mujoco_robots=[robot.robot_model for robot in self.robots],
    #         mujoco_objects=[self.hook, self.cube],
    #     )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize hook tool
        self.hook = HookFrame(
            name="LShapeTool",
            frame_length=0.3,
            frame_height=0.2,
            frame_thickness=0.025,
            hook_height=0.05,
            grip_location=0.1,
            grip_size=(0.02, 0.03),
            use_texture=True,
        )

        # initialize cube (target object to pull)
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cube = BoxObject(
            name="cube",
            size_min=[0.020, 0.020, 0.020],
            size_max=[0.022, 0.022, 0.022],
            rgba=[1, 0, 0, 1],
            material=redwood,
            rng=self.rng,
        )

        # Create placement initializer
        self._get_placement_initializer()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.hook, self.cube],
        )

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds
        """
        # Create placement initializer
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        # Place hook close to robot (within reach)
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="HookSampler",
                mujoco_objects=self.hook,
                x_range=[-0.10, -0.05],
                y_range=[-0.15, 0.15],
                rotation=None,
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )
        )
        
        # Place cube far from robot (out of direct reach, needs hook)
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CubeSampler",
                mujoco_objects=self.cube,
                x_range=[0.20, 0.30],
                y_range=[-0.15, 0.15],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.hook_body_id = self.sim.model.body_name2id(self.hook.root_body)
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # define observables modality
            modality = "object"

            # hook-related observables
            @sensor(modality=modality)
            def hook_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.hook_body_id])

            @sensor(modality=modality)
            def hook_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.hook_body_id]), to="xyzw")

            # cube-related observables
            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")

            sensors = [hook_pos, hook_quat, cube_pos, cube_quat]

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            # gripper to hook position sensor
            sensors += [
                self._get_obj_eef_sensor(full_pf, "hook_pos", f"{arm_pf}gripper_to_hook_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]

            # gripper to cube position sensor
            sensors += [
                self._get_obj_eef_sensor(full_pf, "cube_pos", f"{arm_pf}gripper_to_cube_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]

            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    # def _reset_internal(self):
    #     """
    #     Resets simulation internal configurations.
    #     """
    #     super()._reset_internal()

    #     # Reset all object positions using initializer sampler if we're not directly loading from an xml
    #     if not self.deterministic_reset:
    #         # Sample from the placement initializer for all objects
    #         object_placements = self.placement_initializer.sample()

    #         # Loop through all objects and reset their positions
    #         for obj_pos, obj_quat, obj in object_placements.values():
    #             # Set the hook to lie flat using init_quat
    #             if obj.name == "LShapeTool":
    #                 obj_quat = convert_quat(self.hook.init_quat, to="wxyz")
                
    #             self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
        
    #     # Store initial cube position for reward calculation
    #     self.initial_cube_pos = self.sim.data.body_xpos[self.cube_body_id].copy()

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        
        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            # DON'T call reset() - sample directly
            # self.placement_initializer.reset()  # <-- REMOVE THIS LINE
            
            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                # Set the hook to lie flat using init_quat
                if obj.name == "LShapeTool":
                    obj_quat = convert_quat(self.hook.init_quat, to="wxyz")
                
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
        
        # Store initial cube position for reward calculation
        self.initial_cube_pos = self.sim.data.body_xpos[self.cube_body_id].copy()

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Visualize distance to cube (simple object, not composite)
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube)

    def _check_success(self):
        """
        Check if cube has been successfully pulled into reach.
        Success is defined as the cube being within a certain distance of the robot base.

        Returns:
            bool: True if cube is within reach
        """
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        robot_base_pos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        
        # Check if cube is within reachable distance (e.g., 0.4m from robot base)
        dist_to_robot = np.linalg.norm(cube_pos[:2] - robot_base_pos[:2])
        cube_in_reach = dist_to_robot < 0.4
        
        # Also check that cube is still on the table
        cube_on_table = cube_pos[2] > (self.table_offset[2] - 0.05)
        
        return cube_in_reach and cube_on_table