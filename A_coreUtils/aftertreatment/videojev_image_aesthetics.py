"""使用已编码的 VideoJev 图像特征计算美学分。"""
import json
from math import isclose
from pathlib import Path
from statistics import mean

from .videojev_aesthetic_context import adapt_checks, describe, environmental_portrait_checks
from .videojev_aesthetic_scale import assess



def choice(name, instructions, options):
    return {"name": name, "type": "choice", "instructions": instructions,
            "choices": [{"value": label, "description": desc} for label, desc in options]}

SUBJECT = choice("subject", "这张画面主要在表现什么？看取景、突出程度和视觉焦点，从人、物体、风景中只选一项。", [
    ("人", "重点是一个人或一群人，包括绘制的人物、剪影、背影或人的身体局部。"),
    ("物体", "重点是某个非人的独立对象，如物品、动物、植物、天体、文字、图形或建筑构件。"),
    ("风景", "重点是整处自然或人造环境，如山林、天空、街景、建筑群或室内空间；其中可有人或物点缀。"),
])

EMOTION = choice("emotion", "从主角可见的表情和肢体动作，选择最明显的外在情绪。"
                 "多人时看主要人物，分不出主次且表现不一致时选无法判断。", [
    ("快乐", "明显微笑、大笑、兴奋或庆祝。"), ("悲伤", "流泪、哀伤或沮丧。"),
    ("痛苦", "皱紧五官、痛楚或明显挣扎。"),
    ("愤怒", "怒视、暴怒或明显恼火。"), ("恐惧", "害怕、惊恐或紧张畏缩。"),
    ("惊讶", "突然吃惊、瞪眼或惊愕。"), ("厌恶", "明显嫌弃或排斥的表情。"),
    ("平静", "能看清神态，表情中性或放松。"),
    ("无法判断", "面部太小、被遮挡、背对镜头，或可见线索不足。"),
])

PROFILES = {"人": "人物", "物体": "其他主体", "风景": "景色"}
DETAILED_RUBRICS = json.loads(Path(__file__).with_name("videojev_aesthetic_checks_zh.json").read_text(encoding="utf-8"))


def choice_query(question):
    entries = question["choices"] if question["type"] == "choice" else question["levels"]
    return {"id": question["name"], "task": "choice", "query": question["instructions"],
            "options": [entry["description"] for entry in entries]}


def questions_from_rubrics(rubrics, profile):
    questions = []
    for dimension, criteria in rubrics[profile].items():
        if isinstance(criteria, dict):  # Original one-question-per-dimension format.
            criteria = [{"name": dimension, **criteria}]
        for criterion in criteria:
            questions.append((dimension, {"id": criterion["name"], "task": "choice",
                              "query": criterion["question"], "options": criterion["levels"]}))
    return questions


def score_image(engine, path, rubrics=None, *, contextual=None, environment_aware=True, anchored=None,
                evidence=True, scene_cues=None, face_or_mood=True):
    features, record = engine.encode_image(path)
    return score_features(engine, features, record, rubrics, contextual=contextual,
                          environment_aware=environment_aware, anchored=anchored,
                          evidence=evidence, scene_cues=scene_cues, face_or_mood=face_or_mood)


def score_features(engine, features, record, rubrics=None, *, contextual=None, environment_aware=True,
                   anchored=None, evidence=True, scene_cues=None, face_or_mood=True):
    if scene_cues is None:
        scene_cues = anchored is None and contextual is None and rubrics is None
    if anchored is None:
        anchored = contextual is None and rubrics is None
    if contextual is None:
        contextual = rubrics is None
    if rubrics is None:
        rubrics = DETAILED_RUBRICS
    subject_answer, = engine.model.predict_features(engine.processor, features, record, [choice_query(SUBJECT)])
    subject = SUBJECT["choices"][subject_answer["answer_index"]]["value"]
    profile = PROFILES[subject]
    checks = questions_from_rubrics(rubrics, profile)
    context_fields = {}
    if contextual:
        emotion = choice_query(EMOTION)
        emotion_labels = {c["description"]: c["value"] for c in EMOTION["choices"]}
        if scene_cues:
            emotion["query"] = "只看主角可见的眼、眉、嘴，选择面部表现的情绪。表情中性可选平静，五官线索不足时选无法判断。"
        context_fields, focus = describe(engine, features, record, subject, emotion,
                                         include_face_visibility=scene_cues)
        if subject == "人":
            predicted = context_fields["人物情绪"]
            context_fields["人物情绪"] = emotion_labels.get(predicted, predicted)
        if anchored:
            visible = context_fields.pop("面部可辨", None)
            return assess(engine, features, record, subject, context_fields, evidence=evidence,
                          scene_cues=scene_cues, face_visible=visible, face_or_mood=face_or_mood)
        checks = adapt_checks(checks, subject, context_fields, focus)
        if environment_aware:
            checks = environmental_portrait_checks(checks, subject, context_fields)
    answers = engine.model.predict_features(engine.processor, features, record, [q for _, q in checks])
    grouped = {}
    for (dimension, question), answer in zip(checks, answers, strict=True):
        probs = answer["probabilities"]
        if (answer["id"] != question["id"] or len(probs) != len(question["options"])
                or not all(0 <= p <= 1 for p in probs) or not isclose(sum(probs), 1, abs_tol=1e-5)):
            raise ValueError(f"模型的评分分布无效：{question['id']}")
        grouped.setdefault(dimension, {})[answer["id"]] = {
            "score": round(sum(i * p for i, p in enumerate(probs)) * 100 / (len(probs) - 1), 2),
            "most_likely_description": answer["answer"], "probabilities": probs,
        }
    dimensions = {}
    for name, items in grouped.items():
        if len(items) == 1:
            dimensions[name] = next(iter(items.values()))
        else:
            dimensions[name] = {"score": round(mean(v["score"] for v in items.values()), 2),
                                "checks": items}
    score = round(mean(d["score"] for d in dimensions.values()), 2)
    if contextual:
        form_label = {"人": "人物造型", "物体": "物体造型", "风景": "场景布局"}[subject]
        context_fields["细项评分"] = {"景别取景": grouped["构图"]["景别取景"]["score"],
                                    form_label: grouped["美感"]["主体造型"]["score"]}
        if environment_aware:
            context_fields["细项评分"].update(光影=dimensions["光线"]["score"],
                                             情绪氛围=dimensions["情绪表达"]["score"])
    result = {"subject": subject, "aesthetic_profile": profile, "aesthetic_score": score, "细分": context_fields,
            "aesthetic_score_1_10": int(1.5 + 9 * score / 100), "aesthetic_dimensions": dimensions}
    if any("checks" in d for d in dimensions.values()):
        ordered = sorted(((v["score"], dim, key, v["most_likely_description"]) for dim, items in grouped.items()
                          for key, v in items.items()), key=lambda item: item[0])
        result["review"] = {
            label: {"dimension": dim, "criterion": key, "score": value, "observation": desc}
            for label, (value, dim, key, desc) in (("relative_strength", ordered[-1]), ("relative_weakness", ordered[0]))
        }
    return result
