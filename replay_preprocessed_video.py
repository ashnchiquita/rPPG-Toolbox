import numpy as np
import cv2
import os
import argparse
import glob
from typing import List, Tuple, Optional, Dict
import time
import traceback

class PreprocessedVideoPlayer:
    def __init__(self, data_path: str, data_format: str, data_types: List[str], fps: int = 30):
        """
        Initialize the video player for preprocessed NPY files.
        
        Args:
            data_path: Path to folder containing NPY files
            data_format: Data format (any permutation of N,D,C,H,W like 'NDHWC', 'NDCHW', 'NCHWD', etc.)
            data_types: List of data types in processing order
            fps: Target FPS for playback
        """
        self.data_path = data_path
        self.data_format = data_format.upper()
        self.data_types = data_types
        self.fps = fps
        self.frame_delay = 1.0 / fps
        
        # Parse data format to understand dimension ordering
        self.dim_mapping = self._parse_data_format()
        
    def _parse_data_format(self) -> Dict[str, int]:
        """Parse the data format string to create dimension mapping."""
        valid_dims = set(['N', 'D', 'C', 'H', 'W'])
        format_dims = set(self.data_format)
        
        if not format_dims.issubset(valid_dims):
            raise ValueError(f"Invalid dimensions in format '{self.data_format}'. Valid: {valid_dims}")
        
        # Create mapping from dimension name to axis index
        dim_mapping = {}
        for i, dim in enumerate(self.data_format):
            dim_mapping[dim] = i
            
        print(f"Data format '{self.data_format}' parsed as: {dim_mapping}")
        return dim_mapping
    
    def _transpose_to_standard(self, data: np.ndarray) -> np.ndarray:
        """Transpose data from custom format to standard NCHW format."""
        current_shape = data.shape
        print(f"Original shape: {current_shape} (format without N: {self.data_format.replace('N', '')})")
        
        # NPY files have 4 dimensions (N is always 1 and not stored)
        # Create file format mapping (without N dimension)
        file_format = self.data_format.replace('N', '')
        file_dim_mapping = {}
        for i, dim in enumerate(file_format):
            file_dim_mapping[dim] = i
        
        print(f"File dimension mapping: {file_dim_mapping}")
        
        # Add batch dimension (N=1) at the beginning
        data = np.expand_dims(data, axis=0)
        print(f"Added batch dimension: {data.shape}")
        
        # Create full mapping with N at position 0
        full_dim_mapping = {'N': 0}
        for dim, idx in file_dim_mapping.items():
            full_dim_mapping[dim] = idx + 1  # Shift by 1 due to added N dimension
        
        print(f"Full dimension mapping: {full_dim_mapping}")
        
        # Handle depth dimension - D represents time/frames, not spatial depth
        if 'D' in full_dim_mapping:
            print("Depth/Time dimension detected - this represents frames")
            # Transpose to NDCHW order
            transpose_order = []
            for target_dim in ['N', 'D', 'C', 'H', 'W']:
                if target_dim in full_dim_mapping:
                    transpose_order.append(full_dim_mapping[target_dim])
            
            print(f"Transpose order for NDCHW: {transpose_order}")
            data = np.transpose(data, transpose_order)
            print(f"After transpose to NDCHW: {data.shape}")
            
            # Reshape to (N*D, C, H, W) to treat each frame as a separate sample
            N, D, C, H, W = data.shape
            data = data.reshape(N * D, C, H, W)
            print(f"After reshaping to treat frames as samples: {data.shape}")
        else:
            # No depth dimension, transpose directly to NCHW
            transpose_order = []
            for target_dim in ['N', 'C', 'H', 'W']:
                if target_dim in full_dim_mapping:
                    transpose_order.append(full_dim_mapping[target_dim])
                else:
                    raise ValueError(f"Required dimension '{target_dim}' not found in mapping")
            
            print(f"Transpose order for NCHW: {transpose_order}")
            data = np.transpose(data, transpose_order)
            print(f"After transpose to NCHW: {data.shape}")
        
        return data
    
    def get_npy_files(self, subject_pattern: str = "*") -> List[str]:
        """Get all NPY files matching the subject pattern."""
        # Handle both *_input*.npy and *.npy patterns
        patterns = [
            os.path.join(self.data_path, f"{subject_pattern}_input*.npy"),
        ]
        
        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern))
        
        # Remove duplicates and sort
        files = sorted(list(set(files)))
        return files
    
    def load_and_normalize_data(self, npy_file: str) -> np.ndarray:
        """Load NPY file and normalize data for display."""
        data = np.load(npy_file)
        print(f"Loaded {npy_file}: shape {data.shape}, dtype {data.dtype}")
        
        # Validate shape
        expected_dims = len(self.data_format) - 1  # -1 because N is not in file
        if len(data.shape) != expected_dims:
            raise ValueError(f"Expected {expected_dims}D data, got {len(data.shape)}D: {data.shape}")
        
        # Convert to standard NCHW format
        data = self._transpose_to_standard(data)
        
        # Now data should be (N, C, H, W)
        if len(data.shape) != 4:
            raise ValueError(f"Expected 4D data (NCHW) after processing, got shape: {data.shape}")
        
        return self._process_channels_for_display(data)
    
    def _process_channels_for_display(self, data: np.ndarray) -> np.ndarray:
        """Process channels based on data types for proper display."""
        N, C, H, W = data.shape
        print(f"Processing {C} channels for display")
        
        # Handle DiffNormalized case (6 channels)
        if 'DiffNormalized' in self.data_types and C == 6:
            # DiffNormalized typically contains [R, G, B, R_diff, G_diff, B_diff]
            # For display, we can use the first 3 channels (original RGB)
            display_data = data[:, 3:6, :, :]
            print("Using first 3 channels from DiffNormalized data")
        elif C >= 3:
            # Use first 3 channels as RGB
            display_data = data[:, :3, :, :]
            print(f"Using first 3 channels out of {C} total channels")
        elif C == 1:
            # Handle grayscale - replicate to 3 channels
            display_data = np.repeat(data, 3, axis=1)
            print("Converting grayscale to RGB by replication")
        else:
            raise ValueError(f"Cannot handle {C} channels for display")
        
        # Normalize data for display (0-255 range)
        display_data = self._normalize_for_display(display_data)
        
        # Convert from (N, C, H, W) to (N, H, W, C) for OpenCV
        display_data = np.transpose(display_data, (0, 2, 3, 1))
        
        return display_data.astype(np.uint8)
    
    def _normalize_for_display(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to 0-255 range for display."""
        print(f"Data range before normalization: [{np.min(data):.3f}, {np.max(data):.3f}]")
        
        # Handle different normalization based on data types
        if 'Standardized' in self.data_types:
            # Standardized data typically has mean=0, std=1
            # Convert back to 0-1 range assuming original was normalized
            data = (data * 0.5) + 0.5  # Rough denormalization
            print("Applied inverse standardization")
        elif 'DiffNormalized' in self.data_types:
            # DiffNormalized might have different ranges
            print("DiffNormalized data detected")
        elif 'Raw' in self.data_types:
            # Raw data might already be in 0-255 range
            if np.max(data) > 1.0:
                print("Raw data appears to be in 0-255 range")
                return np.clip(data, 0, 255)
            else:
                print("Raw data appears to be in 0-1 range")
        
        # Min-max normalization to 0-255
        data_min = np.min(data)
        data_max = np.max(data)
        
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min) * 255
        else:
            data = np.zeros_like(data)
        
        print(f"Data range after normalization: [{np.min(data):.3f}, {np.max(data):.3f}]")
        return np.clip(data, 0, 255)
    
    def play_video(self, npy_file: str, window_name: Optional[str] = None):
        """Play a single NPY video file."""
        if window_name is None:
            window_name = f"Video: {os.path.basename(npy_file)}"
        
        window_created = False
        try:
            frames = self.load_and_normalize_data(npy_file)
            print(f"Playing {frames.shape[0]} frames at {self.fps} FPS")
            print("Controls: 'q'=quit, 'p'=pause/resume, 'r'=restart, '←'/'→'=seek")
            
            # Try different OpenCV window flags for compatibility
            try:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Use WINDOW_NORMAL instead
            except AttributeError:
                cv2.namedWindow(window_name)  # Fallback to default
            
            window_created = True
            
            frame_idx = 0
            paused = False
            
            while True:
                if not paused:
                    if frame_idx >= len(frames):
                        frame_idx = 0  # Loop video
                    
                    frame = frames[frame_idx]
                    
                    # Add frame info overlay
                    info_text = f"{frame_idx+1}/{len(frames)} | Format: {self.data_format} | Types: {','.join(self.data_types)}"
                    cv2.putText(frame, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.imshow(window_name, frame)
                    frame_idx += 1
                
                # Handle keyboard input
                key = cv2.waitKey(int(self.frame_delay * 1000)) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif key == ord('r'):
                    frame_idx = 0
                    paused = False
                    print("Restarted")
                elif key == 81:  # Left arrow
                    frame_idx = max(0, frame_idx - 10)
                    print(f"Seek to frame {frame_idx}")
                elif key == 83:  # Right arrow
                    frame_idx = min(len(frames) - 1, frame_idx + 10)
                    print(f"Seek to frame {frame_idx}")
                    
        except Exception as e:
            print(f"Error playing video: {e}")
            traceback.print_exc()
        finally:
            # Only destroy window if it was created successfully
            if window_created:
                try:
                    cv2.destroyWindow(window_name)
                except:
                    pass  # Ignore errors when destroying window
                
                # Alternative cleanup
                try:
                    cv2.destroyAllWindows()
                except:
                    pass
    
    def play_subject_videos(self, subject_pattern: str = "*"):
        """Play all videos for a subject pattern."""
        files = self.get_npy_files(subject_pattern)
        
        if not files:
            print(f"No files found matching pattern: {subject_pattern}")
            return
        
        print(f"Found {len(files)} video files:")
        for i, file in enumerate(files):
            print(f"{i}: {os.path.basename(file)}")
        
        for file in files:
            print(f"\nPlaying: {os.path.basename(file)}")
            self.play_video(file)
            
            # Ask if user wants to continue
            response = input("Continue to next video? (y/n/q): ").lower()
            if response == 'q':
                break
            elif response == 'n':
                continue

def main():
    parser = argparse.ArgumentParser(description="Replay preprocessed NPY video files with custom data formats")
    parser.add_argument("data_path", help="Path to folder containing NPY files")
    parser.add_argument("--data_format", default="NDHWC", 
                       help="Data format (any permutation of N,D,C,H,W like 'NDHWC', 'NDCHW', 'NCHWD', etc.)")
    parser.add_argument("--data_types", nargs="+", 
                       default=["Raw"],
                       choices=["Raw", "DiffNormalized", "Standardized"],
                       help="Data types in processing order")
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS")
    parser.add_argument("--subject", default="*", help="Subject pattern (e.g., 'subject1' or '*')")
    parser.add_argument("--single_file", help="Play a single specific NPY file")
    
    args = parser.parse_args()
    
    # Validate data format
    valid_dims = set(['N', 'D', 'C', 'H', 'W'])
    if not set(args.data_format.upper()).issubset(valid_dims):
        print(f"Error: Invalid data format '{args.data_format}'. Must contain only: {valid_dims}")
        return
    
    # Check if N is in format (it should be for complete specification)
    if 'N' not in args.data_format.upper():
        print(f"Warning: N dimension not specified in format '{args.data_format}'. Adding N at the beginning.")
        args.data_format = 'N' + args.data_format.upper()
    
    print(f"Using data format: {args.data_format}")
    print(f"Note: NPY files are expected to have 4 dimensions (N=1 is not stored in file)")
    
    player = PreprocessedVideoPlayer(
        data_path=args.data_path,
        data_format=args.data_format,
        data_types=args.data_types,
        fps=args.fps
    )
    
    if args.single_file:
        player.play_video(args.single_file)
    else:
        player.play_subject_videos(args.subject)

if __name__ == "__main__":
    main()
