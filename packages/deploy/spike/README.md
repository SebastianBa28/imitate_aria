# Milestone 0: check the robot Mac before building on it

Run these on the Apple Silicon Mac, in order. Each prints `OK` or `FAIL`/`STOP` with the reason.
None of them moves the robot.

```bash
cd packages/deploy
pixi install -e hw            # 1. does the environment solve and install on osx-arm64?
pixi run -e hw spike-env      #    every import should say ok
```

2. **ROS over Ethernet to the Franka machine.** Same `ROS_DOMAIN_ID` on both (e.g. `export ROS_DOMAIN_ID=7`).
   Copy `spike/ros_ping.py` to the Franka machine and run `python3 ros_ping.py pong` there (its own ROS sourced). Then on the Mac:
   ```bash
   pixi run -e hw spike-ping ping     # round trip should be a few ms
   pixi run -e hw spike-ping arm      # with the controller running: its pose topic and rate
   ```
   If nothing arrives: macOS firewall (System Settings > Network > Firewall: allow incoming for python),
   both machines on the same subnet, and `ROS_AUTOMATIC_DISCOVERY_RANGE` not set to `LOCALHOST`.

3. **RealSense.** D435 on a USB 3 port with a USB 3 cable:
   ```bash
   pixi run -e hw spike-realsense
   ```
   On recent macOS, librealsense often needs root for USB: `sudo -E env PATH=$PATH python spike/realsense_check.py`
   from inside `pixi shell -e hw`. If only sudo works, the RealSense node will have to run that way too.
   If `pyrealsense2-macosx` does not install, report the pixi error: the fallback is building librealsense from source.

4. **ORCA hand.** Copy this hand's `calibration.yaml` from the rwr_system machine
   (`src/hand_control/orca/configs/orcahand_v2/right/calibration.yaml`) to `config/orca/right/` (gitignored), then:
   ```bash
   pixi run -e hw spike-orca                         # or: ... spike-orca --port /dev/tty.usbserial-XXXX
   ```
   If the pixi solve fails on `orca_core` because of its numpy requirement, install it the rwr_system way:
   remove it from `pixi.toml`, then `pixi run -e hw python -m pip install --no-deps orca_core==0.5.2`
   plus its runtime dependencies (`dynamixel-sdk`, `pyserial`).

Write the results down (versions, rates, what needed sudo): they decide the next steps of the plan.
