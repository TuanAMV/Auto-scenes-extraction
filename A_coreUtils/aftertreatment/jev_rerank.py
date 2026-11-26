"""两条搜索流程共用的 VideoJev 单图推理。模型在发布版嵌入式 Python 子进程中加载。"""
from collections import defaultdict
from contextlib import contextmanager, redirect_stdout
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Lock

PROJECT = Path(__file__).resolve().parents[2]
_gpu_lock = Lock()


def score_prefix(similarity, item, *, include_similarity=True):
    if 'aesthetic_score' in item:
        score = item['aesthetic_score']
        aesthetic = '未评分' if score is None else f'{score:.2f}'
        return (f'{similarity:.4f}_' if include_similarity else '') + aesthetic + '_'
    return f'{similarity:.4f}_' if include_similarity else ''


@contextmanager
def extracted_frames(targets, cache_dir):
    """按原视频的绝对帧号抽图；不同目录下的同名视频独立缓存。"""
    from PIL import Image
    from A_coreUtils.search.reranker_frame_extractor import RerankerFrameExtractor

    by_video = defaultdict(set)
    for video, frame in targets:
        if video and frame is not None:
            by_video[video].add(frame)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    paths = {}
    with TemporaryDirectory(dir=cache_dir) as temp:
        extractor = RerankerFrameExtractor(output_resolution='384', cache_dir=temp)
        for idx, (video, frames) in enumerate(by_video.items()):
            try:
                for frame, bgr in extractor.iter_memory_frames(video, sorted(frames)):
                    path = Path(temp, f'{idx}_{frame}.jpg')
                    Image.fromarray(bgr[:, :, ::-1]).save(path, quality=extractor.jpeg_quality)
                    paths[video, frame] = str(path.resolve())
            except (OSError, RuntimeError, ValueError) as exc:
                print(f'[Jev-rerank] 无法读取 {video} 的请求帧：{exc}')
        yield paths


class JevReranker:
    def __init__(self):
        self.process = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.process is None:
            return
        try:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        finally:
            self.process.stdout.close()
            self.process = None
            _gpu_lock.release()

    def evaluate(self, path, questions, *, use_chinese=True, aesthetics=False,
                 min_matches=0, threshold=0.5):
        if not questions and not aesthetics:
            return {}
        if self.process is None:
            python = sys.executable
            _gpu_lock.acquire()
            try:
                self.process = subprocess.Popen(
                    [str(python), '-u', '-m', 'A_coreUtils.aftertreatment.jev_rerank'],
                    cwd=PROJECT, text=True, encoding='utf-8', stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    env={**os.environ, 'PYTHONUTF8': '1', 'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_PROGRESS_BARS': '1'},
                )
            except BaseException:
                _gpu_lock.release()
                raise
        request = dict(path=str(path), questions=questions, use_chinese=use_chinese,
                       aesthetics=aesthetics, min_matches=min_matches, threshold=threshold)
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'VideoJev 推理进程已退出（{self.process.poll()}），请查看服务日志')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(f"Jev-rerank 推理失败：{result['error']}")
        return result


def _evaluate(engine, request):
    features, record = engine.encode_image(request['path'])
    options = ['是', '否'] if request['use_chinese'] else ['Yes', 'No']
    queries = [dict(id=key, task='choice', query=prompt, options=options)
               for key, prompt in request['questions'].items()]
    scores = {}
    if queries:
        answers = engine.model.predict_features(engine.processor, features, record, queries)
        for query, answer in zip(queries, answers, strict=True):
            probability = float(answer['probabilities'][0])
            if answer['id'] != query['id'] or not 0 <= probability <= 1:
                raise ValueError(f"模型返回了无效的答案：{answer}")
            scores[query['id']] = probability
    result = {'scores': scores}
    if request['aesthetics'] and sum(p >= request['threshold'] for p in scores.values()) >= request['min_matches']:
        from .videojev_image_aesthetics import score_features
        result['aesthetic_score'] = score_features(engine, features, record)['aesthetic_score']
    return result


def _serve(out):
    import torch
    from .videojev_model import MODEL_PATH, VideoJevQA

    engine = None
    for line in sys.stdin:
        try:
            if engine is None:
                engine = VideoJevQA(MODEL_PATH)
            with torch.inference_mode():
                result = _evaluate(engine, json.loads(line))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            result = {'error': str(exc)}
        print(json.dumps(result, allow_nan=False), file=out, flush=True)


if __name__ == '__main__':
    protocol = sys.stdout
    with redirect_stdout(sys.stderr):
        _serve(protocol)
