import cv2
import numpy as np
import torch
import math
from typing import List, Tuple, Optional

class GPUResizer:
    def __init__(self, device_id: int = 0):
        """
        Initialize GPU Resizer
        Args:
        device_id: GPU device ID
        """
        self.device_id = device_id
        cv2.cuda.setDevice(device_id)
        self.stream = cv2.cuda.Stream()
        
        # Get GPU info
        self.gpu_name = torch.cuda.get_device_name(device_id)
        self.total_memory = torch.cuda.get_device_properties(device_id).total_memory
        
    def get_available_memory(self) -> int:
        """Get current available GPU memory (in bytes)"""
        torch.cuda.synchronize()
        return torch.cuda.mem_get_info(self.device_id)[0]  # Return available memory
    
    def estimate_batch_size(self, frame_shape: Tuple[int, ...], new_shape: Tuple[int, int]) -> int:
        """
        Estimate a safe batch size
         Args:
             frame_shape: Shape of the input frame (height, width, channels)
            new_shape: Target dimensions (new_height, new_width)
         Returns:
             The suggested batch size
        """
        # Estimate the GPU memory usage for a single image (including input and output).
        input_size = np.prod(frame_shape) * 4  # 4 bytes per float32
        output_size = new_shape[0] * new_shape[1] * frame_shape[2] * 4
        single_frame_memory = (input_size + output_size) * 1.5  # Extra 50%
        
        # Get available memory
        available_memory = self.get_available_memory()
        
        # Calculate a safe batch size (using 80% of available memory)
        safe_memory = available_memory * 0.8
        batch_size = max(1, int(safe_memory / single_frame_memory))
        
        return min(batch_size, 100)  # Max Limit is 100
    
    def resize_batch(self, 
                    frames: np.ndarray, 
                    new_width: int, 
                    new_height: int,
                    show_progress: bool = False) -> Optional[np.ndarray]:
        """
        Resize a batch of images using the GPU
        Args:
            frames: Input frames (N, H, W, C)
            new_width: Target width
            new_height: Target height
        show_progress: Whether to display progress
        Returns:
            The resized frames
        """
        try:
            if frames.size == 0:
                raise ValueError("Empty input frames")
            
            # Calculate a suitable batch size
            batch_size = self.estimate_batch_size(
                frames[0].shape, 
                (new_height, new_width)
            )
            
            total_frames = len(frames)
            processed_frames = []
            
            
            if show_progress:
                from tqdm import tqdm
                pbar = tqdm(total=total_frames, desc="Resizing frames")
            
            
            for start_idx in range(0, total_frames, batch_size):
                end_idx = min(start_idx + batch_size, total_frames)
                current_batch = frames[start_idx:end_idx]
                
                try:
                    
                    gpu_frames = []
                    for frame in current_batch:
                        gpu_mat = cv2.cuda_GpuMat()
                        gpu_mat.upload(frame)
                        gpu_frames.append(gpu_mat)
                    
                    
                    batch_output = []
                    with self.stream:
                        for gpu_frame in gpu_frames:
                            resized = cv2.cuda.resize(
                                gpu_frame, 
                                (new_width, new_height),
                                interpolation=cv2.INTER_AREA
                            )
                            cpu_resized = resized.download()
                            batch_output.append(cpu_resized)
                    
                    
                    for gpu_frame in gpu_frames:
                        gpu_frame.release()
                    
                    processed_frames.extend(batch_output)
                    
                    if show_progress:
                        pbar.update(len(current_batch))
                        
                except cv2.error as e:
                    print(f"OpenCV GPU error in batch {start_idx}-{end_idx}: {str(e)}")
                    continue
                except Exception as e:
                    print(f"Unexpected error in batch {start_idx}-{end_idx}: {str(e)}")
                    continue
                
            if show_progress:
                pbar.close()
            
            if not processed_frames:
                return None
                
            return np.stack(processed_frames)
            
        except Exception as e:
            print(f"Error in resize_batch: {str(e)}")
            return None
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stream.synchronize()
        cv2.cuda.resetDevice()

def resize_video_frames_gpu(frames: np.ndarray, 
                       new_width: int,
                       new_height: int,
                       device_id: int = 0,
                       show_progress: bool = False) -> Optional[np.ndarray]:
    """
    Main function to resize video frames
    Args:
        frames: Input frames (N, H, W, C)
        new_width: Target width
        new_height: Target height
        device_id: GPU device ID
        show_progress: Whether to display progress
    Returns:
        The resized frames
    """
    if frames is None or frames.size == 0:
        return None
        
    with GPUResizer(device_id) as resizer:
        return resizer.resize_batch(
            frames, 
            new_width, 
            new_height,
            show_progress
        )


if __name__ == "__main__":
    
    frames = np.random.randint(0, 255, (100, 1080, 1920, 3), dtype=np.uint8)
    
    
    resized_frames = resize_video_frames(
        frames,
        scale_factor=0.5,
        device_id=0,
        show_progress=True
    )
    
    if resized_frames is not None:
        print(f"Input shape: {frames.shape}")
        print(f"Output shape: {resized_frames.shape}")
