
Project Title: Cricut Mat Image Rectification & Precision Grid Aligner
Target Output: Python script/application that takes distorted photos of artwork on a 12 \times 12\text{ inch} Cricut mat, auto-orients, auto-detects the 1-inch grid network, interpolates occluded nodes, provides an interactive GUI verification overlay, and exports a high-resolution 3600 \times 3600\text{ px} (300 DPI) square PNG ready for Cricut Design Space.

Key Requirements & Constraints

1. Orientation Auto-Correction: Detect the hanging slot cutout on the outer mat margin to determine top-side orientation and rotate the image (0^\circ, 90^\circ, 180^\circ, 270^\circ) so the physical top faces UP.
2. Line & Grid Node Detection: Detect the 13 horizontal and 13 vertical primary 1-inch grid lines using OpenCV line transforms and HSV color filtering for the green background.
3. Occlusion Resilience: Line extrapolation and Thin Plate Splines (TPS) / RANSAC homography must infer grid nodes covered by artwork, paper, or masking tape.
4. Interactive Verification GUI: Display an interactive overlay showing detected vs. interpolated nodes, allowing manual drag-and-nudge fine-tuning before final processing.
5. Piecewise Affine Warping: Warp each 1 \times 1\text{ inch} block independently to eliminate camera lens barrel/pincushion distortion and physical mat bending.
6. 300 DPI Export: Save a perfectly square 3600 \times 3600\text{ px} image cropped strictly to the 12 \times 12\text{ inch} grid boundary.

Modular Architecture & Deliverables

Module 1: Preprocessing & Mat Orientation (orientation.py)

⚬ Task: Isolate the green mat using HSV thresholding (cv2.inRange). Locate the outer border contours and detect the non-green geometry corresponding to the hanging hole cutout.
⚬ Deliverable: A function normalize_orientation(image: np.ndarray) -> np.ndarray that rotates the image matrix so the hanging hole is positioned at the top-center.

Module 2: Line Detection & RANSAC Grid Initialization (grid_detector.py)

⚬ Task: Extract white grid lines using adaptive thresholding and cv2.HoughLinesP. Group detected lines into horizontal and vertical sets. Compute full geometric line intersections (13 \times 13 matrix = 169 nodes).
⚬ Deliverable: A function detect_grid_nodes(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray] returning an array of candidate 2D point coordinates and a boolean mask indicating high-confidence detected vs. low-confidence/occluded nodes.

Module 3: Mesh Interpolation & Deformation Engine (mesh_warper.py)

⚬ Task: Fit a global Homography using RANSAC on high-confidence nodes. Use scipy.interpolate.Rbf or Thin Plate Splines (TPS) to smoothly estimate the coordinates of hidden/occluded interior nodes based on surrounding visible node displacement vectors.
⚬ Deliverable: A class GridMesh that holds the 13 \times 13 topological mesh, provides methods to recalculate missing nodes, and executes skimage.transform.PiecewiseAffineTransform to warp the raw image into a 3600 \times 3600\text{ px} array.

Module 4: Interactive Verification GUI (gui.py)

⚬ Task: Build a GUI using OpenCV callbacks (cv2.setMouseCallback) or PyQt/Matplotlib. Render the preliminary unwarped image with a cyan vector grid over 1 \times 1\text{ inch} squares. Draw visible nodes as blue dots and interpolated nodes as orange dots. Allow dragging nodes with real-time mesh updating.
⚬ Deliverable: An interactive UI window with keyboard shortcuts (R to reset, Space/Enter to confirm, arrow keys to nudge active selection).

Module 5: CLI / Pipeline Runner (main.py)

⚬ Task: Glue modules 1–4 together into a single executable command-line interface. Set up image loading, execution flow, DPI metadata insertion (using PIL / PieEXIF), and output file saving.
⚬ Deliverable: Command python main.py --input input_photo.jpg --output rectified_artwork.png --dpi 300.

Test:
Test images for verification can be found in ./Test folder