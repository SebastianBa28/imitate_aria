"""Topic and service names between the robot nodes, the policy and the dashboard.

Arm topics keep the contract of the external Franka controller (as in rwr_system): it subscribes to
ARM_POSE_CMD and publishes ARM_POSE, both geometry_msgs/PoseStamped in frame ARM_FRAME.
"""

# hand (sensor_msgs/JointState, radians, names = hand.JOINT_NAMES)
HAND_STATE = "/right/hand/joint_states"
HAND_COMMAND = "/right/hand/command"
HAND_ESTOP = "/right/hand/estop"            # std_srvs/SetBool, true = release torque and disconnect
HAND_TORQUE = "/right/hand/set_torque"      # std_srvs/SetBool

# arm (geometry_msgs/PoseStamped)
ARM_POSE = "/right/arm/end_effector_pose"
ARM_POSE_CMD = "/right/arm/end_effector_pose_cmd"
ARM_FRAME = "world"

# camera (RealSense D435)
CAMERA_COLOR = "/camera/color/image_raw"           # sensor_msgs/Image rgb8
CAMERA_DEPTH = "/camera/depth/image_raw"           # sensor_msgs/Image 16UC1, mm, aligned to color
CAMERA_INFO = "/camera/color/camera_info"          # sensor_msgs/CameraInfo

# policy
POLICY_ACTION_HAND = "/policy/action_hand"         # JointState, published in preview and live
POLICY_ACTION_ARM = "/policy/action_arm"           # PoseStamped, published in preview and live
POLICY_METRICS = "/policy/metrics"                 # std_msgs/String, JSON (see policy_node)
POLICY_STATE = "/policy/state"                     # std_msgs/String, JSON, latched
POLICY_COMMAND = "/policy/command"                 # std_msgs/String: start|stop|reset|home|live|preview; result shows in POLICY_STATE
