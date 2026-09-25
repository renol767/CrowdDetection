import cv2
import numpy as np
import os
import random

def create_synthetic_crowd_video(output_path: str, duration_sec: int = 15, fps: int = 25):
    """
    Creates a synthetic drone-perspective surveillance video of a plaza with 3 crowd clusters
    (simulating DPR RI front gate plaza: Zona A in center, Zona B on left, Zona C on right).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 1280, 720
    total_frames = duration_sec * fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Generating synthetic crowd video to {output_path} ({total_frames} frames)...")

    # Generate crowd people particles
    # Cluster A: ~70 people in center
    # Cluster B: ~40 people in left
    # Cluster C: ~35 people in right
    num_a = 75
    num_b = 45
    num_c = 40
    
    people = []
    
    for _ in range(num_a):
        people.append({
            "x": random.uniform(width * 0.38, width * 0.62),
            "y": random.uniform(height * 0.46, height * 0.80),
            "vx": random.uniform(-0.8, 0.8),
            "vy": random.uniform(-0.5, 0.5),
            "h": random.randint(38, 55),
            "w": random.randint(16, 24),
            "color_shirt": (random.randint(40, 240), random.randint(40, 240), random.randint(40, 240)),
            "zone": "A"
        })

    for _ in range(num_b):
        people.append({
            "x": random.uniform(width * 0.10, width * 0.32),
            "y": random.uniform(height * 0.45, height * 0.75),
            "vx": random.uniform(-0.7, 0.7),
            "vy": random.uniform(-0.5, 0.5),
            "h": random.randint(38, 55),
            "w": random.randint(16, 24),
            "color_shirt": (random.randint(40, 240), random.randint(40, 240), random.randint(40, 240)),
            "zone": "B"
        })

    for _ in range(num_c):
        people.append({
            "x": random.uniform(width * 0.68, width * 0.90),
            "y": random.uniform(height * 0.46, height * 0.75),
            "vx": random.uniform(-0.7, 0.7),
            "vy": random.uniform(-0.5, 0.5),
            "h": random.randint(38, 55),
            "w": random.randint(16, 24),
            "color_shirt": (random.randint(40, 240), random.randint(40, 240), random.randint(40, 240)),
            "zone": "C"
        })

    for f in range(total_frames):
        # Create tactical drone camera background (asphalt road, barriers, landmark building)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (45, 48, 52) # Dark asphalt plaza

        # Top background: City skyline & green dome building silhouette
        cv2.rectangle(frame, (0, 0), (width, int(height * 0.38)), (95, 110, 120), -1)
        # Green dome landmark (DPR dome)
        cv2.ellipse(frame, (width // 2, int(height * 0.35)), (180, 80), 0, 180, 360, (70, 140, 100), -1)
        # Building pillars / gate barrier
        cv2.rectangle(frame, (int(width * 0.25), int(height * 0.34)), (int(width * 0.75), int(height * 0.40)), (30, 32, 35), -1)
        # Road markings
        for x_line in range(50, width, 120):
            cv2.line(frame, (x_line, int(height * 0.42)), (x_line + 40, int(height * 0.42)), (180, 180, 180), 2)

        # Sort people by y position for natural depth rendering
        people.sort(key=lambda p: p["y"])

        # Render people silhouettes
        for p in people:
            # Update position
            p["x"] += p["vx"]
            p["y"] += p["vy"]

            # Bounce off bounds
            if p["zone"] == "A":
                if p["x"] < width * 0.35 or p["x"] > width * 0.65: p["vx"] *= -1
                if p["y"] < height * 0.44 or p["y"] > height * 0.82: p["vy"] *= -1
            elif p["zone"] == "B":
                if p["x"] < width * 0.08 or p["x"] > width * 0.34: p["vx"] *= -1
                if p["y"] < height * 0.42 or p["y"] > height * 0.78: p["vy"] *= -1
            elif p["zone"] == "C":
                if p["x"] < width * 0.66 or p["x"] > width * 0.92: p["vx"] *= -1
                if p["y"] < height * 0.44 or p["y"] > height * 0.78: p["vy"] *= -1

            # Perspective scale
            scale = 0.6 + (p["y"] / height) * 0.6
            pw = int(p["w"] * scale)
            ph = int(p["h"] * scale)
            px = int(p["x"])
            py = int(p["y"])

            # Shadow
            cv2.ellipse(frame, (px, py), (pw, pw // 3), 0, 0, 360, (20, 20, 20), -1)

            # Head
            head_r = int(pw * 0.45)
            cv2.circle(frame, (px, py - ph + head_r), head_r, (170, 140, 110), -1)

            # Torso / Clothes
            torso_top = py - ph + (head_r * 2)
            cv2.rectangle(frame, (px - pw // 2, torso_top), (px + pw // 2, py - ph // 3), p["color_shirt"], -1)

            # Legs
            cv2.line(frame, (px - pw // 4, py - ph // 3), (px - pw // 4, py), (30, 30, 40), max(1, pw // 4))
            cv2.line(frame, (px + pw // 4, py - ph // 3), (px + pw // 4, py), (30, 30, 40), max(1, pw // 4))

        # Add banners / flags in crowd (like screenshot)
        cv2.rectangle(frame, (int(width * 0.42), int(height * 0.62)), (int(width * 0.58), int(height * 0.65)), (230, 230, 235), -1)
        cv2.putText(frame, "SUARA RAKYAT", (int(width * 0.44), int(height * 0.645)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 20, 20), 2)

        writer.write(frame)

    writer.release()
    print(f"Generated synthetic crowd demo video successfully: {output_path}")
    return output_path

if __name__ == "__main__":
    create_synthetic_crowd_video("uploads/demo_crowd.mp4")
