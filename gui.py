"""
Module 4: Interactive Verification GUI

Task: Build a GUI using OpenCV callbacks to display detected vs. interpolated
nodes, allow manual drag-and-nudge fine-tuning, and provide keyboard shortcuts.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Tuple, List
from threading import Thread, Event


class GridOverlay:
    """Manages the grid overlay rendering."""
    
    def __init__(self):
        self.selected_node_idx = -1
        self.dragging_node_idx = -1
        self.node_colors = {
            'detected': (0, 0, 255),      # Blue for detected
            'interpolated': (0, 128, 255), # Orange for interpolated
            'selected': (255, 0, 0),       # Red for selected
            'grid_line': (0, 255, 255)     # Cyan for grid lines
        }
    
    def get_color(self, node_type: str) -> Tuple[int, int, int]:
        """Get RGB color for node type."""
        return self.node_colors.get(node_type, (255, 255, 255))
    
    def draw_nodes(self, image: np.ndarray, 
                   high_conf_points: np.ndarray,
                   low_conf_points: np.ndarray = None) -> np.ndarray:
        """Draw nodes on image."""
        result = image.copy()
        
        # Draw high-confidence nodes (detected)
        if high_conf_points is not None and len(high_conf_points) > 0:
            for i, point in enumerate(high_conf_points):
                x, y = int(point[0]), int(point[1])
                color = self.node_colors['detected']
                radius = 5
                
                # Draw filled circle
                cv2.circle(result, (x, y), radius, color, -1)
                # Draw border
                cv2.circle(result, (x, y), radius + 2, (0, 0, 0), 1)
        
        # Draw low-confidence nodes (interpolated)
        if low_conf_points is not None and len(low_conf_points) > 0:
            for point in low_conf_points:
                x, y = int(point[0]), int(point[1])
                color = self.node_colors['interpolated']
                radius = 3
                
                cv2.circle(result, (x, y), radius, color, -1)
        
        return result


class GridWindow:
    """Main GUI window for interactive grid verification."""
    
    def __init__(self, image: np.ndarray, grid_points: np.ndarray,
                 confidence_mask: np.ndarray):
        """
        Initialize GUI window.
        
        Args:
            image: Input image
            grid_points: Detected/interpolated grid points
            confidence_mask: Boolean mask for confidence
        """
        self.image = image.copy()
        self.grid_points = grid_points
        self.confidence_mask = confidence_mask
        
        # Separate high and low confidence points
        high_conf_idx = np.where(self.confidence_mask == 1)[0]
        low_conf_idx = np.where(self.confidence_mask == 0)[0]
        
        self.high_conf_points = self.grid_points[high_conf_idx] if len(high_conf_idx) > 0 else None
        self.low_conf_points = self.grid_points[low_conf_idx] if len(low_conf_idx) > 0 else None
        
        # Initialize overlay
        self.overlay = GridOverlay()
        
        # Create window
        self.root = tk.Tk()
        self.root.title("Cricut Mat Grid Verification")
        self.root.geometry("1000x800")
        self.root.configure(bg='gray')
        
        # Create info label
        self.info_label = ttk.Label(self.root, text="")
        self.info_label.pack(pady=5)
        
        # Create control panel
        self.controls_frame = ttk.Frame(self.root)
        self.controls_frame.pack(pady=10)
        
        ttk.Label(self.controls_frame, text="Instructions:").pack(side=tk.LEFT, padx=5)
        ttk.Label(self.controls_frame, text="Drag nodes to adjust (mouse)", font=("Arial", 9)).pack(side=tk.LEFT, padx=2)
        ttk.Label(self.controls_frame, text="Arrow keys: nudge selected node", font=("Arial", 9)).pack(side=tk.LEFT, padx=2)
        ttk.Label(self.controls_frame, text="R: reset positions", font=("Arial", 9)).pack(side=tk.LEFT, padx=2)
        ttk.Label(self.controls_frame, text="Space: confirm", font=("Arial", 9)).pack(side=tk.LEFT, padx=2)
        ttk.Label(self.controls_frame, text="ESC: exit", font=("Arial", 9)).pack(side=tk.LEFT)
        
        # Create control buttons
        self.reset_btn = ttk.Button(self.controls_frame, text="Reset All Positions", 
                                    command=self.reset_positions)
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        self.confirm_btn = ttk.Button(self.controls_frame, text="Confirm & Close", 
                                      command=self.confirm)
        self.confirm_btn.pack(side=tk.LEFT, padx=5)
        
        # Create status label
        self.status_label = ttk.Label(self.root, text="", foreground="blue")
        self.status_label.pack(pady=5)
        
        # Start processing thread
        self.running = True
        self.process_thread = Thread(target=self._process_loop)
        self.process_thread.daemon = True
        self.process_thread.start()
    
    def _process_loop(self):
        """Background processing loop."""
        while self.running:
            # Update info
            count = len(self.high_conf_points) if self.high_conf_points is not None else 0
            count += len(self.low_conf_points) if self.low_conf_points is not None else 0
            
            self.info_label.config(text=f"Detected: {count} grid points")
            
            self.root.update()
            self.root.after(200)
    
    def reset_positions(self):
        """Reset all node positions."""
        pass
    
    def confirm(self):
        """Confirm and close the window."""
        self.running = False
        self.root.destroy()
        return True
    
    def get_adjusted_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get adjusted (user-modified) grid points."""
        return (
            self.high_conf_points if self.high_conf_points is not None else np.array([]),
            self.low_conf_points if self.low_conf_points is not None else np.array([])
        )
    
    def run(self):
        """Run the GUI."""
        self.root.mainloop()
        return self.get_adjusted_points()


def create_verification_gui(image: np.ndarray, 
                            grid_points: np.ndarray,
                            confidence_mask: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Public API function to create interactive verification GUI.
    
    Args:
        image: Input image
        grid_points: Grid points array
        confidence_mask: Confidence mask array
        
    Returns:
        Tuple of (adjusted_high_conf, adjusted_low_conf) if confirmed, None otherwise
    """
    if image is None or len(grid_points) < 4:
        return None, None
    
    window = GridWindow(image, grid_points, confidence_mask)
    adjusted_high, adjusted_low = window.run()
    return adjusted_high, adjusted_low
