"""Milestone 0, step 2: can this machine talk ROS 2 with the Franka control machine over Ethernet?

    python spike/ros_ping.py pong          # on the Franka machine (plain python3 with its ROS sourced)
    python spike/ros_ping.py ping          # on the Mac: prints round-trip times
    python spike/ros_ping.py arm           # on the Mac: checks the controller's arm pose topic and rate

Both machines need the same ROS_DOMAIN_ID and must not set ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST.
Uses only std_msgs/geometry_msgs, so it needs nothing from this repo.
"""
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Float64

mode = sys.argv[1] if len(sys.argv) > 1 else "ping"
rclpy.init()
node = Node(f"spike_{mode}")

if mode == "pong":
    pub = node.create_publisher(Float64, "/spike/pong", 10)
    node.create_subscription(Float64, "/spike/ping", lambda m: pub.publish(m), 10)
    print("echoing /spike/ping -> /spike/pong, Ctrl+C to stop")
    rclpy.spin(node)

elif mode == "ping":
    rtts = []
    node.create_subscription(Float64, "/spike/pong", lambda m: rtts.append(time.perf_counter() - m.data), 10)
    pub = node.create_publisher(Float64, "/spike/ping", 10)
    node.create_timer(0.1, lambda: pub.publish(Float64(data=time.perf_counter())))
    t_end = time.time() + 10
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not rtts:
        print("FAIL: no pong in 10 s (domain id, firewall, multicast, cable?)")
    else:
        ms = sorted(r * 1e3 for r in rtts)
        print(f"OK: {len(ms)}/100 replies, round trip median {ms[len(ms) // 2]:.2f} ms, max {ms[-1]:.2f} ms")

elif mode == "arm":
    stamps = []
    node.create_subscription(PoseStamped, "/right/arm/end_effector_pose", lambda m: stamps.append((time.time(), m)), 10)
    t_end = time.time() + 5
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)
    if len(stamps) < 2:
        print("FAIL: no /right/arm/end_effector_pose in 5 s (is the controller running?)")
    else:
        hz = (len(stamps) - 1) / (stamps[-1][0] - stamps[0][0])
        p = stamps[-1][1].pose.position
        print(f"OK: {hz:.1f} Hz, frame {stamps[-1][1].header.frame_id!r}, position {p.x:.3f} {p.y:.3f} {p.z:.3f}")
else:
    sys.exit(f"unknown mode {mode!r}: use pong, ping or arm")

node.destroy_node()
rclpy.try_shutdown()
