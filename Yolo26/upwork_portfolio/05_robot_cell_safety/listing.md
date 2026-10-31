# Upwork portfolio entry: Robot Cell Safety Monitor

## Project title
Human-Robot Safety Monitoring with Pose Estimation & Open-Vocab AI

## Your role
Computer Vision Engineer: system design, safety logic, implementation

## Project description
Built a vision safety monitor for robot work cells. YOLO26-pose tracks workers' hands and feet, and YOLOE open-vocabulary segmentation finds robots from text prompts, with no robot dataset needed. It measures the separation from every body keypoint to the robot in metres and issues SAFE/SLOW/STOP speed commands with hysteresis in the style of ISO/TS 15066, plus floor zones, a person-down alarm, fail-safe handling when the robot is occluded, and an event log for PLC/MQTT. About 9 FPS on CPU.

## Skills (tags)
Computer Vision · Industrial Automation · Pose Estimation · Robotics · Python

## Media (upload in this order)
1. `cover.png`
2. `demo.mp4`: UR10 cell (SAFE → STOP → SLOW → STOP → SAFE) plus a cobot assembly line
3. `screenshot_1.jpg` … `screenshot_3.jpg`

## Key results to mention
- Robots segmented zero-shot from text prompts; negative prompts stop people being labelled as robots
- Hand-level separation distance; escalates immediately, releases after 1 s
- Full event log: t=9.0 s SAFE→STOP (0.45 m), 24.1 s STOP→SLOW (0.8 m), …

## Longer text
Robots are leaving their cages: cobots, AMRs and soon humanoids share floors with people. Laser scanners see only an ankle-height slice and produce no analytics. This monitor adds hand-level awareness from a single camera. Worker keypoints from YOLO26-pose are compared against a robot mask from YOLOE-26, segmented by text prompt, using a distance transform. Pixel distances become metres using each worker's torso length. Floor zones, a hysteresis state machine and a person-down detector drive a recommended robot speed (100/30/0 %) and a timestamped event log ready for PLC, OPC UA or MQTT. It's designed as a decision-support and near-miss analytics layer alongside certified safety devices.

Code: github.com/DavitGadyan/Computer_Vision/tree/master/Yolo26/projects/05_robot_cell_safety
