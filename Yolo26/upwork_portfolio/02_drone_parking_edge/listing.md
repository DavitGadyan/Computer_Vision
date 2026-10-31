# Upwork portfolio entry: Drone Smart Parking (Edge AI)

## Project title
Drone & Edge AI Parking Occupancy Detection (YOLO26-OBB, Pi / Jetson)

## Your role
Computer Vision / Edge AI Engineer: detection pipeline, stabilization, edge deployment

## Project description
Built a parking-occupancy system for drone or mast-camera video using YOLO26 oriented bounding boxes, which give tight rotated boxes for top-down cars. It compensates for drone drift with ORB + RANSAC homography, so the slot map stays aligned and false events dropped from 32 to 3. Debounced slot states feed a live dashboard and JSON/MQTT events. Exported to NCNN (Raspberry Pi), IMX500 (Pi AI Camera), OpenVINO and TensorRT (Jetson) for low-cost edge deployment.

## Skills (tags)
Computer Vision · Edge AI · YOLO · Raspberry Pi · NVIDIA Jetson

## Media (upload in this order)
1. `cover.png`
2. `demo.mp4`: 20 s drone-view demo with live dashboard
3. `screenshot_1.jpg`, `screenshot_2.jpg`

## Key results to mention
- 34 angled slots monitored, ~28 vehicles per frame detected with rotated boxes
- Drift compensation: 32 → 3 slot events in 35 s (no false flicker)
- Edge exports: ONNX, NCNN, OpenVINO benchmarked on CPU; IMX500 and TensorRT deployment paths

## Longer text
In-ground parking sensors cost $200–400 per space. One drone pass or a single pole camera can cover hundreds. I used YOLO26-OBB, pre-trained on aerial imagery, for rotated vehicle boxes. Slot rows are defined parametrically: six numbers describe a whole row of angled spaces. Frames are registered to a reference view so the slot map follows the drifting drone, and an asymmetric debounce absorbs missed detections of dark cars in shadow. The NMS-free YOLO26 graph exports cleanly to NCNN for Raspberry Pi, IMX500 for the Raspberry Pi AI Camera's on-sensor NPU, and TensorRT for Jetson. Only a few bytes of JSON per minute leave the device.

Code: github.com/DavitGadyan/Computer_Vision/tree/master/Yolo26/projects/02_drone_parking_edge
