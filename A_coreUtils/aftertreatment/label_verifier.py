# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存
"""
MiniCPM-V-4-6 标签验证模块 — mini_rerank（Reranker 平替）

对 CLIP 筛选出的每类最佳标签做存在性校验：
输入中间帧 + {类别: 标签}，输出 {类别: 是否匹配}。

与 reranker 不同：reranker 做图文相似度精排，mini_rerank 直接判断标签是否在画面中。

Usage:
    from A_coreUtils.aftertreatment.label_verifier import LabelVerifier
    lv = LabelVerifier()
    result = lv.verify(frame_bgr, {"主体": "男人", "场景": "城市"})
    # {"主体": True, "场景": False}
"""

import json
import numpy as np

# 缓存: 从 logic_keywords.json 加载的验证 prompt 模板
_VERIFY_TEMPLATES_CACHE = None


def _load_verify_templates():
    """从 logic_keywords.json 加载 MiniCPM 验证 prompt 模板。"""
    global _VERIFY_TEMPLATES_CACHE
    if _VERIFY_TEMPLATES_CACHE is not None:
        return _VERIFY_TEMPLATES_CACHE
    try:
        from path_resolver import PathResolver
        r = PathResolver()
        path = str(r.project_root / 'logic_keywords.json')
        with open(path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        _VERIFY_TEMPLATES_CACHE = cfg.get('其他参数设置', {}).get('MiniCPM验证Prompt', {})
    except Exception:
        pass
    if not _VERIFY_TEMPLATES_CACHE:
        _VERIFY_TEMPLATES_CACHE = {
            'zh': {
                '主体': '以{val}为主体',
                '动作': '主体正在{val}',
                '场景': '以{val}为背景',
                '情绪': '画面渲染{val}情绪',
                '_instruction': '判断这张图片是否包含以下描述，只输出一行JSON不要解释:',
                '_format': '格式: {{{pairs}}}',
                '_yes_no': '"是/否"',
            },
            'en': {
                '主体': 'a {val} as the main subject',
                '动作': 'the subject is {val}',
                '场景': 'with a {val} backdrop',
                '情绪': 'with a {val} mood',
                '_instruction': 'Determine whether this image matches the following descriptions. Output only one line of JSON without explanation:',
                '_format': 'Format: {{{pairs}}}',
                '_yes_no': '"yes/no"',
            },
        }
    return _VERIFY_TEMPLATES_CACHE


class LabelVerifier:
    """基于 MiniCPM-V-4-6 的标签存在性验证器。"""

    def __init__(self):
        self._analyzer = None

    @property
    def analyzer(self):
        if self._analyzer is None:
            from A_coreUtils.aftertreatment.shot_analyzer import ShotAnalyzer
            self._analyzer = ShotAnalyzer()
            self._analyzer.load_model()
        return self._analyzer

    def verify(self, frame_bgr: "np.ndarray", labels: dict, use_chinese: bool = True) -> dict:
        """判断视频中间帧是否包含指定标签。

        Args:
            frame_bgr: BGR numpy 数组 (H, W, 3)
            labels: {类别: 标签名}，中文或英文取决于 use_chinese
            use_chinese: 标签是中文还是英文，控制 prompt 语言和判断关键词

        Returns:
            {类别: True/False}，True 表示标签在画面中存在
        """
        if not labels:
            return {}

        import torch
        from PIL import Image
        import cv2

        analyzer = self.analyzer
        categories = list(labels.keys())

        # 构建 prompt（中/英文），从 logic_keywords.json 读取模板
        tmpl_cfg = _load_verify_templates()
        lang_key = 'zh' if use_chinese else 'en'
        lang_cfg = tmpl_cfg.get(lang_key, {})
        templates = {k: v for k, v in lang_cfg.items() if not k.startswith('_')}
        yes_no = lang_cfg.get('_yes_no', '"是/否"' if use_chinese else '"yes/no"')
        instruction = lang_cfg.get('_instruction',
            '判断这张图片是否包含以下描述，只输出一行JSON不要解释:' if use_chinese
            else 'Determine whether this image matches the following descriptions. Output only one line of JSON without explanation:')
        format_tpl = lang_cfg.get('_format', '格式: {{{pairs}}}' if use_chinese else 'Format: {{{pairs}}}')
        lines = []
        for cat in categories:
            tpl = templates.get(cat, "- {cat}: {val}")
            line = tpl.format(cat=cat, val=labels[cat])
            lines.append(line)
        cat_pairs = ', '.join('"%s": %s' % (c, yes_no) for c in categories)
        prompt = (
            instruction + "\n"
            + "\n".join(lines) + "\n"
            + format_tpl.format(pairs=cat_pairs)
        )

        # 图片推理
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        msgs = [{"role": "user", "content": [
            {"type": "image", "image": pil_img},
            {"type": "text", "text": prompt},
        ]}]

        inputs = analyzer._processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(analyzer.device)

        with torch.no_grad():
            generated_ids = analyzer._model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                eos_token_id=analyzer._processor.tokenizer.eos_token_id,
            )
        response_ids = generated_ids[0][inputs["input_ids"].shape[1]:]
        text = analyzer._processor.tokenizer.decode(response_ids, skip_special_tokens=True)

        # 解析
        return self._parse(text, categories)

    def _parse(self, text: str, categories: list) -> dict:
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise ValueError(
                "MiniCPM 回复不含有效 JSON: %s" % text[:200]
            )
        raw = text[start:end + 1]
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                "MiniCPM JSON 解析失败: %s\n原文: %s" % (e, raw[:200])
            ) from e
        result = {}
        for cat in categories:
            val = str(parsed.get(cat, "")).strip()
            result[cat] = "是" in val or "yes" in val.lower()
        return result

    def unload(self):
        if self._analyzer is not None:
            self._analyzer.unload_model()
            self._analyzer = None
