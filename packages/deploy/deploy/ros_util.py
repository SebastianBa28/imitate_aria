"""Message helpers (no cv_bridge: images are converted with numpy) and shared QoS profiles."""
import numpy as np
from geometry_msgs.msg import PoseStamped
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, JointState

# images: drop frames rather than queue them
SENSOR_QOS = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE)
# state/info topics a late subscriber must still receive
LATCHED_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def joint_state_msg(stamp, names, position, velocity=None):
    msg = JointState()
    msg.header.stamp = stamp
    msg.name = list(names)
    msg.position = [float(x) for x in position]
    if velocity is not None:
        msg.velocity = [float(x) for x in velocity]
    return msg


def pose_msg(stamp, frame, position, quat):
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = frame
    p, q = msg.pose.position, msg.pose.orientation
    p.x, p.y, p.z = (float(v) for v in position)
    q.x, q.y, q.z, q.w = (float(v) for v in quat)
    return msg


def pose_from_msg(msg):
    p, q = msg.pose.position, msg.pose.orientation
    return np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w])


_ENCODINGS = {"rgb8": (np.uint8, 3), "bgr8": (np.uint8, 3), "mono8": (np.uint8, 1), "16UC1": (np.uint16, 1)}


def image_msg(stamp, frame, array, encoding):
    dtype, channels = _ENCODINGS[encoding]
    array = np.ascontiguousarray(array, dtype=dtype)
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame
    msg.height, msg.width = array.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = msg.width * channels * array.itemsize
    msg.data = array.tobytes()
    return msg


def image_to_numpy(msg):
    dtype, channels = _ENCODINGS[msg.encoding]
    a = np.frombuffer(msg.data, dtype=dtype)
    row = msg.step // np.dtype(dtype).itemsize
    a = a.reshape(msg.height, row)[:, : msg.width * channels]
    return a.reshape(msg.height, msg.width, channels) if channels > 1 else a.reshape(msg.height, msg.width)


def camera_info_msg(stamp, frame, width, height, fx, fy, cx, cy, coeffs=(0.0,) * 5, model="plumb_bob"):
    msg = CameraInfo()
    msg.header.stamp = stamp
    msg.header.frame_id = frame
    msg.width, msg.height = int(width), int(height)
    msg.distortion_model = model
    msg.d = [float(c) for c in coeffs]
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg
