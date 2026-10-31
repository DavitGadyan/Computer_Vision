# Upwork portfolio kit

One folder per project, ready to paste into **Upwork → Profile → Portfolio → Add project**.

| Folder | Upwork title |
|---|---|
| [`01_workout_monitoring`](01_workout_monitoring/listing.md) | AI Fitness Coach: Real-Time Rep Counting & Form Feedback (Pose AI) |
| [`02_drone_parking_edge`](02_drone_parking_edge/listing.md) | Drone & Edge AI Parking Occupancy Detection (YOLO26-OBB, Pi / Jetson) |
| [`03_solar_panel_inspection`](03_solar_panel_inspection/listing.md) | Solar Panel Defect Inspection with Custom-Trained YOLO26 (EL Images) |
| [`04_traffic_segmentation_speed_anpr`](04_traffic_segmentation_speed_anpr/listing.md) | Traffic AI: Vehicle Segmentation, Speed Estimation & Plate Recognition |
| [`05_robot_cell_safety`](05_robot_cell_safety/listing.md) | Human-Robot Safety Monitoring with Pose Estimation & Open-Vocab AI |

Each folder contains:

| File | Use it for |
|---|---|
| `listing.md` | **Title** (≤70 chars), **Your role**, **Description** (≤600 chars), **Skills**, key results, and a longer write-up for the project page or proposals |
| `cover.png` | Portfolio thumbnail, 1600×1200 (4:3), titled |
| `demo.mp4` | Annotated demo video (H.264, < 10 MB). Upload it directly or put it on YouTube/Loom and paste the link |
| `screenshot_*.jpg` | Extra gallery images |

Tips
- Put the strongest result in the first line of the description, e.g. "plate model mAP50 0.97" or "32 → 3 false events".
- The same media works for LinkedIn posts and proposal attachments.
- To regenerate media after re-running a demo, use `python scripts/make_assets.py ...` (see each project README).
