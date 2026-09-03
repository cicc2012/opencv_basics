This repo covers OpenCV foundations, hands-on prototype code, and camera calibration steps to guide you through building the vision system.

In this lab, you will build and test the vision pipeline on the Raspberry Pi 5 using Python 3 and OpenCV.

---

## 1. Computer Vision Foundations & HSV Color Space

### What is HSV Space?

By default, images are represented in **RGB** (Red, Green, Blue). While RGB is great for displays, it is very sensitive to lighting changes. A shadow falling across white tile or black tape changes the $R$, $G$, and $B$ values drastically, breaking simple color filters.

**HSV** separates color information into three distinct channels:

* **Hue ($H$):** The pure color family (e.g., Red, Yellow, Green, Blue). Range in OpenCV: $0\text{--}179$.
* **Saturation ($S$):** The intensity or purity of the color (grey vs. vibrant). Range: $0\text{--}255$.
* **Value ($V$):** The brightness of the color (dark vs. light). Range: $0\text{--}255$.

### Why is HSV useful for detecting black tape?

Black tape lacks vibrant color (low Saturation) and reflects very little light (low Value). Filtering in HSV space allows you to isolate black tape by targeting **low $V$ (Brightness)** values, making your detection resilient to ambient room lighting variations.

* **Intuitiveness**: Artists and designers find it easier to pick a base hue and adjust its shade or tint using saturation and value rather than balancing three separate red, green, and blue light intensities.
* **Computer Vision**: As noted in discussions on platforms like Stack Exchange, developers often rely on the OpenCV library's HSV space for object tracking and color-based segmentation because **hue remains relatively stable under varying shadow and lighting conditions**.

---

### Prototype Exercise: Interactive HSV Trackbar Tuning

Run this script on your Pi using a live camera feed or an image of your floor tape. Adjust the sliders until the tape appears **pure white** and everything else turns **pure black**.

```python
import cv2
import numpy as np

def nothing(x):
    pass

# Create a window for sliders
cv2.namedWindow("HSV Tuner")
cv2.createTrackbar("Low H", "HSV Tuner", 0, 179, nothing)
cv2.createTrackbar("High H", "HSV Tuner", 179, 179, nothing)
cv2.createTrackbar("Low S", "HSV Tuner", 0, 255, nothing)
cv2.createTrackbar("High S", "HSV Tuner", 255, 255, nothing)
cv2.createTrackbar("Low V", "HSV Tuner", 0, 255, nothing)      # Black tape: keep low
cv2.createTrackbar("High V", "HSV Tuner", 80, 255, nothing)     # Black tape: limit upper brightness

# Initialize camera feed (or load image via cv2.imread)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Convert BGR image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. Get current positions of trackbars
    l_h = cv2.getTrackbarPos("Low H", "HSV Tuner")
    u_h = cv2.getTrackbarPos("High H", "HSV Tuner")
    l_s = cv2.getTrackbarPos("Low S", "HSV Tuner")
    u_s = cv2.getTrackbarPos("High S", "HSV Tuner")
    l_v = cv2.getTrackbarPos("Low V", "HSV Tuner")
    u_v = cv2.getTrackbarPos("High V", "HSV Tuner")

    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])

    # 3. Create binary mask (White = inside range, Black = outside)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # 4. Optional: Clean up noise using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    cv2.imshow("Original Frame", frame)
    cv2.imshow("Filtered Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

```

---

## 2. Demos: Edge vs. Vertex vs. Shape Detection

Depending on your preference, there are three main algorithmic paths to estimate the parking spot's open edge. Compare these prototype approaches to choose the best strategy for your team.

```
       [ Input Binary Mask ]
                 │
   ┌─────────────┼─────────────┐
   ▼             ▼             ▼
[ Approach A ] [ Approach B ] [ Approach C ]
  Prob. Hough   Contour Poly   Intersected 
  Line Edges   Approximation     Vertices

```

---

### Approach A: Edge Detection (Probabilistic Hough Lines)

Finds all straight line segments independently using `cv2.HoughLinesP`.

```python
import cv2
import numpy as np

# Assuming 'mask' is your binary image from Step 1
edges = cv2.Canny(mask, 50, 150)
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=40, minLineLength=30, maxLineGap=10)

canvas = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Draw each detected line segment
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Hough Line Detection", canvas)

```

* **Pros:** Works even if the black tape contour is partially broken or faint.
* **Cons:** Requires manual logic to group adjacent line segments together.

---

### Approach B: Shape Detection (Contour Polygon Approximation)

Treats the entire shape as a single continuous perimeter loop and simplifies it using `cv2.approxPolyDP`.

```python
import cv2

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
canvas = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

for cnt in contours:
    if cv2.contourArea(cnt) > 500:  # Ignore tiny noise
        # Calculate perimeter
        peri = cv2.arcLength(cnt, True)
        # Approximate shape (epsilon = 2% of perimeter)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # A 3-sided tape marker usually yields 4 primary vertices
        if len(approx) == 4:
            cv2.drawContours(canvas, [approx], -1, (0, 255, 0), 3)
            for pt in approx:
                x, y = pt[0]
                cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)

cv2.imshow("Polygon Approximation", canvas)

```

* **Pros:** Highly structured output (returns an ordered list of key corner points).
* **Cons:** Requires a clean, unbroken contour around the whole tape boundary.

---

### Approach C: Vertex Intersections (Shared vs. Open Vertices)

Extracts segment endpoints and classifies them: endpoints that connect to two lines are **interior corners**, while endpoints belonging to only one line are **open vertices**.

```python
import cv2
import numpy as np

def get_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Extract lines from Hough transform
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=30, maxLineGap=10)
endpoints = []

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        endpoints.append((x1, y1))
        endpoints.append((x2, y2))

# Group endpoints that are close to each other (< 15 pixels radius)
clusters = []
for pt in endpoints:
    matched = False
    for c in clusters:
        if get_distance(pt, c['center']) < 15:
            c['points'].append(pt)
            matched = True
            break
    if not matched:
        clusters.append({'center': pt, 'points': [pt]})

# Classify vertices based on connection count
canvas = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
open_vertices = []

for c in clusters:
    # A cluster containing multiple segment endpoints is an interior corner
    if len(c['points']) >= 2:
        cv2.circle(canvas, c['center'], 6, (255, 0, 0), -1) # Blue = Shared Corner
    else:
        # Single endpoints mark the open boundary!
        cv2.circle(canvas, c['center'], 6, (0, 0, 255), -1) # Red = Open Vertex
        open_vertices.append(c['center'])

# Draw the open edge line between isolated endpoints
if len(open_vertices) == 2:
    cv2.line(canvas, open_vertices[0], open_vertices[1], (0, 255, 255), 2)

cv2.imshow("Vertex Intersection Analysis", canvas)

```

---

## 3. Camera Calibration (Checkerboard Method)

Camera lenses introduce optical **barrel distortion**, making straight lines on the floor look curved. Calibration calculates a **Camera Matrix** and **Distortion Coefficients** to undistort images before running vision algorithms.

```
       +---------------------------------------------+
       |             CHECKERBOARD BOARD              |
       |  +---+---+---+---+---+---+---+---+---+---+  |
       |  |   |   |   |   |   |   |   |   |   |   |  |
       |  +---+---+---+---+---+---+---+---+---+---+  |
       |  |   |   |   |   |   |   |   |   |   |   |  |
       |  +---+---+---+---+---+---+---+---+---+---+  |
       |             8 x 6 Inner Corners             |
       +---------------------------------------------+

```

### Preparing a Checkerboard Target

1. **Print Pattern:** Download a standard $9 \times 7$ square checkerboard pattern (which contains $8 \times 6$ **inner grid corners**).
2. **Mount Rigidly:** Glue the paper flat onto a rigid piece of cardboard or wood. *Any warps or bends in the board will ruin the calibration.*
3. **Measure Grid:** Use a precision ruler to measure the exact square size in millimeters (e.g., $25\text{ mm}$).

---

### Step-by-Step Calibration Script

Capture 15–20 images of the checkerboard from different angles and distances before running this script:

```python
import cv2
import numpy as np
import glob

# Define grid size (number of INSIDE corners, not total squares)
CHECKERBOARD = (8, 6) 
SQUARE_SIZE_MM = 25.0  # Measured width of one square

# Prepare 3D object points (0,0,0), (25,0,0), (50,0,0)...
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE_MM

objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane

images = glob.glob('calibration_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        # Refine corner locations for high precision
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

# Run OpenCV Camera Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("--- CALIBRATION COMPLETE ---")
print("Camera Matrix (Intrinsics):\n", mtx)
print("Distortion Coefficients:\n", dist)

# Save calibration results to a file for use in your parking project
np.savez("camera_calib.npz", matrix=mtx, dist=dist)

```

---

## 4. First-Time OpenCV Warm-Up & Best Practices

Before jumping into full robot integration, review these fundamental tips:

### A. Coordinate System Alignment

* OpenCV image coordinates start at **$(0,0)$ in the top-left corner**.
* $X$ increases as you move **Right**.
* $Y$ increases as you move **Down**.

```
(0,0) ───────> +X (Columns)
  │
  │
  v
 +Y (Rows)

```

### B. Resolution vs. Processing Latency

High camera resolutions ($1920 \times 1080$) severely degrade frame rates on mobile processors.

* For vision-guided vehicles, **downsample frames** to **$640 \times 480$** or **$320 \times 240$**.
* Processing at lower resolutions reduces memory overhead and allows real-time execution ($30\text{+} \text{ FPS}$).

### C. Useful OpenCV Utility Snippets

#### 1. Resizing Frame for Speed

```python
frame_small = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

```

#### 2. Overlaying Text & Target Markers

```python
# Draw a crosshair at center frame
cv2.drawMarker(frame, (320, 240), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

# Display real-time status text
cv2.putText(frame, "TARGET DETECTED", (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

```

#### 3. FPS Performance Monitor

```python
import time

prev_time = time.time()

while True:
    # Processing pipeline here...
    
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    
    print(f"Vision Processing Speed: {fps:.1f} FPS")

```
