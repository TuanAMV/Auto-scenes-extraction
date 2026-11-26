"""本地 VideoJev-4B 单帧和多帧选择题推理。"""
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[2] / "models/MCG-NJU--VideoJev-4B/snapshots/master"


class VideoJevQA:
    def __init__(self, path):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            str(path), trust_remote_code=True, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to("cuda").prepare_for_inference()

    def encode_frames(self, frames, fps=6):
        # 与官方 prepare_video 使用相同的原生视频处理器，直接传入公共帧，避免二次采样。
        from transformers.video_utils import VideoMetadata

        n = len(frames)
        video = self.processor.video_processor
        meta = VideoMetadata(total_num_frames=n, fps=fps, frames_indices=list(range(n)))
        inputs = video(videos=[frames], do_sample_frames=False, video_metadata=[meta],
                       size={"shortest_edge": video.size["shortest_edge"], "longest_edge": n * 640**2},
                       return_tensors="pt")
        grid = inputs["video_grid_thw"]
        centers = self.processor._calculate_timestamps(list(range(n)), fps, video.temporal_patch_size)
        if len(centers) != int(grid[0, 0]):
            raise ValueError("VideoJev 时间网格与公共帧数不符")
        record = {"grid_thw": grid[0].tolist(), "tokens": int(grid.prod()) // video.merge_size**2,
                  "frames": n, "fps": fps, "score_times": centers}
        features = self.model.encode_video(inputs["pixel_values_videos"], grid)
        return features, record

    def encode_image(self, path):
        import numpy as np
        from PIL import Image, ImageOps

        with Image.open(path) as image:
            pixels = np.array(ImageOps.exif_transpose(image).convert("RGB"))
        return self.encode_frames(pixels[None])
