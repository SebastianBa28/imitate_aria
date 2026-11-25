"""Milestone 0, step 1: does the pixi environment work on this machine? Prints versions; moves nothing."""
import importlib
import os
import platform

print(f"platform: {platform.platform()} ({platform.machine()}), python {platform.python_version()}")
for mod in ("rclpy", "numpy", "cv2", "yaml", "fastapi", "uvicorn", "imitate_common", "deploy",
            "serial", "orca_core", "pyrealsense2"):
    try:
        m = importlib.import_module(mod)
        print(f"  ok   {mod} {getattr(m, '__version__', '')}")
    except Exception as e:  # report every missing piece, not just the first
        print(f"  FAIL {mod}: {e.__class__.__name__}: {e}")
print("ROS_DISTRO:", os.environ.get("ROS_DISTRO"), "| RMW:", os.environ.get("RMW_IMPLEMENTATION", "(default)"),
      "| ROS_DOMAIN_ID:", os.environ.get("ROS_DOMAIN_ID", "(0)"))
