# Upwork portfolio entry: AI Workout Monitor

## Project title
AI Fitness Coach: Real-Time Rep Counting & Form Feedback (Pose AI)

## Your role
Computer Vision Engineer: model selection, algorithm design, implementation, demo

## Project description
Built a real-time AI workout monitor on a single camera using YOLO26 pose estimation. It tracks several athletes at once, recognizes the exercise automatically (squat, push-up, curl), counts reps with a smoothed state machine and gives live form cues such as "go deeper" and "keep body straight". It ignores mirror reflections and exports a per-session JSON report with form scores. Runs at 15 FPS on a laptop CPU with no GPU, and exports to TFLite/CoreML for mobile apps.

## Skills (tags)
Computer Vision · Pose Estimation · YOLO · Python · OpenCV

## Media (upload in this order)
1. `cover.png`: portfolio thumbnail (4:3)
2. `demo.mp4`: 25 s demo (push-ups, home squats, coached gym squats)
3. `screenshot_1.jpg` … `screenshot_3.jpg`

## Key results to mention
- 4/4 push-ups and 7/7 squats counted correctly on test clips, with exercise auto-detected
- Multi-person: coach and trainee tracked separately; mirror reflections filtered out
- ~15 FPS end-to-end on CPU (i9 laptop), 1280×720

## Longer text (for the project page or a proposal)
Fitness apps and connected-gym hardware need reliable rep counting and form analysis from an ordinary camera. I built a pipeline on Ultralytics YOLO26-pose and ByteTrack that measures joint angles (knee, elbow, hip, body line) from the more visible side of each athlete. A per-exercise hysteresis state machine counts reps and separates full-range reps from partial ones. Form rules produce human-readable coaching cues. Adding new exercises is a configuration change, not a new model. The system runs in real time on CPU and can be deployed on-device (TFLite/CoreML/NCNN), so video never leaves the user's phone.

Code: github.com/DavitGadyan/Computer_Vision/tree/master/Yolo26/projects/01_workout_monitoring
