# Step-by-Step Guide: Setting Up CAN Interface and ROS 2 CAN Bridge

This guide walks you through the process of setting up your CAN interface, verifying its operation, launching ROS 2 nodes for CAN data reception, and inspecting CAN messages in ROS 2.

---

## 1. Bring Up the CAN Interface

First, set up your CAN interface (`can0`) with a bitrate of 500,000 bps:

```bash
sudo ip link set can0 up type can bitrate 500000
```

**Note:**  
- Replace `can0` with your actual CAN device name if different.
- This command requires `iproute2` utilities and root privileges.

---

## 2. Verify the CAN Interface

Check if the CAN interface is up and functioning by monitoring incoming CAN frames:

```bash
candump can0
```

**You should see CAN messages if the bus is active.**  
Press `Ctrl+C` to stop monitoring.

---

## 3. Source Your ROS 2 Workspace

Navigate to your ROS 2 workspace and source the environment:

```bash
cd ~/your_ros2_workspace  # Replace with your actual workspace path
source install/setup.bash
```

---

## 4. Launch the Radar Receiver Node

Start the radar receiver using the provided ROS 2 launch file:

```bash
ros2 launch off_highway_radar receveri_launch.py
```

---

## 5. Launch the CAN Bridge Node (New Terminal)

Open a **new terminal**, source your workspace again, and launch the CAN bridge:

```bash
cd ~/your_ros2_workspace  # If not already there; replace as necessary
source install/setup.bash
ros2 launch ros2_socketcan socket_can_bride.launch.xml
```

*(Make sure the launch file name is correct; `socket_can_bride.lanunh.xml` seems like a typo.  
Should be `socket_can_bridge.launch.xml` or similar.)*

---

## 6. List All ROS 2 Topics (New Terminal)

Open **another new terminal**, source your workspace, and list all available ROS 2 topics:

```bash
cd ~/your_ros2_workspace
source install/setup.bash
ros2 topic list
```

You should see topics related to CAN, such as `/from_can_bus`.

---

## 7. Echo CAN Messages from ROS 2 Topic

To view CAN messages being published on a specific topic (e.g., `/from_can_bus`):

```bash
ros2 topic echo /from_can_bus
```

You will see the real-time CAN messages as ROS 2 messages in your terminal.
