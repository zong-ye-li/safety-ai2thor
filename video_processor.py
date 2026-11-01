from __future__ import annotations
import base64
import math
from pathlib import Path
import numpy as np
import numpy.typing as npt


class VideoProcessor:
    """Utility class for processing video frames from the AI2-THOR simulator."""

    def __init__(self):
        pass

    def inspection_canvas(self, frame, width=448, height=448, *, encoded=False):
        """Reconstruct exactly the displayed canvas;never access native state."""
        import hashlib
        if encoded:
            import io
            from PIL import Image
            data=base64.b64decode(self.encode_frames([frame],width,height)[0],validate=True)
            with Image.open(io.BytesIO(data)) as image:
                canvas=np.array(image.convert('RGB'))
            source={'transport':'decoded_recorded_jpeg_input',
                    'encoded_sha256':hashlib.sha256(data).hexdigest()}
        else:
            canvas=self.native_frames([frame],width,height)[0]
            source={'transport':'lossless_fitted_RGB_input'}
        source.update(displayed_size_wh=[width,height],
                      canvas_rgb_sha256=hashlib.sha256(canvas.tobytes()).hexdigest())
        return canvas,source

    def native_frames(self, frames, width=448, height=448):
        """Fit RGB into the requested bounds with centered black padding, never stretching."""
        import cv2
        if min(width,height)<1:
            raise ValueError("Image bounds must be positive")
        result=[]
        for frame in frames:
            if frame.dtype!=np.uint8 or frame.ndim!=3 or frame.shape[2]!=3 or min(frame.shape[:2])<1:
                raise ValueError("Expected nonempty uint8 RGB frames")
            source_h,source_w=frame.shape[:2]
            scale=min(width/source_w,height/source_h)
            fit_w=max(1,min(width,round(source_w*scale)))
            fit_h=max(1,min(height,round(source_h*scale)))
            resized=cv2.resize(frame,(fit_w,fit_h))
            canvas=np.zeros((height,width,3),dtype=np.uint8)
            left,top=(width-fit_w)//2,(height-fit_h)//2
            canvas[top:top+fit_h,left:left+fit_w]=resized
            result.append(canvas)
        return result

    def encode_frames(
        self,
        frames: list[npt.NDArray[np.uint8]],
        width: int = 448,
        height: int = 448,
    ) -> list[str]:
        """Resize RGB frames and encode them into base64 JPEG strings.

        Args:
            frames: List of RGB uint8 numpy arrays.
            width: Target width for resizing.
            height: Target height for resizing.

        Returns:
            List of base64-encoded JPEG strings.
        """
        encoded_frames = []
        import cv2
        for resized in self.native_frames(frames,width,height):
            # Convert RGB to BGR for OpenCV encoding
            bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            # JPEG encode with quality 85
            success, buffer = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if success:
                b64_str = base64.b64encode(buffer).decode('utf-8')
                encoded_frames.append(b64_str)
        return encoded_frames

    def downsample(
        self,
        frames: list[np.ndarray],
        source_fps: float,
        target_fps: float = 2.0,
    ) -> list[np.ndarray]:
        """Downsample the frame list to match a target frame rate.

        Args:
            frames: List of frames to downsample.
            source_fps: Current frame rate of the input frames.
            target_fps: Desired frame rate.

        Returns:
            Downsampled list of frames.
        """
        if not frames:
            return []
        step = round(source_fps / target_fps)
        step = max(1, step)
        return frames[::step]

    def export_mp4(
        self,
        frames: list[np.ndarray],
        dest_path: str,
        fps: float = 2.0,
    ) -> dict:
        """Export a list of RGB frames to an MP4 video file.

        Args:
            frames: List of RGB uint8 numpy arrays.
            dest_path: Path to the destination file.
            fps: Frame rate of the output video.
        """
        if not frames:
            raise ValueError("Video requires at least one frame")

        dest_dir = Path(dest_path).parent
        dest_dir.mkdir(parents=True, exist_ok=True)

        h, w = frames[0].shape[:2]
        if any(frame.shape!=(h,w,3) or frame.dtype!=np.uint8 for frame in frames):
            raise ValueError("All video frames must have identical native RGB shape")
        import cv2
        if Path(dest_path).exists():
            raise FileExistsError(dest_path)
        if not frames or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video requires frames and a positive frame rate")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        encoded_w,encoded_h=w+w%2,h+h%2
        writer = cv2.VideoWriter(dest_path, fourcc, fps, (encoded_w, encoded_h))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open MP4 writer: {dest_path}")

        try:
            for frame in frames:
                # RGB to BGR for OpenCV VideoWriter
                bgr = frame[:, :, ::-1]
                if encoded_h!=h or encoded_w!=w:
                    bgr=cv2.copyMakeBorder(bgr,0,encoded_h-h,0,encoded_w-w,cv2.BORDER_CONSTANT,value=0)
                writer.write(bgr)
        finally:
            writer.release()
        decoded=0
        capture=cv2.VideoCapture(dest_path)
        try:
            while True:
                ok,frame=capture.read()
                if not ok:
                    break
                if frame.shape[:2]!=(encoded_h,encoded_w):
                    raise RuntimeError("Encoded video dimensions differ from the declared frame")
                decoded+=1
        finally:
            capture.release()
        if decoded!=len(frames):
            raise RuntimeError(f"Encoded video contains {decoded} frames; expected {len(frames)}")
        return {"native_width":w,"native_height":h,"encoded_width":encoded_w,"encoded_height":encoded_h,
                "padding_right":encoded_w-w,"padding_bottom":encoded_h-h,"frames":decoded,"fps":fps,
                "sampling":"one frame per recorded event; playback is not inference wall time"}
