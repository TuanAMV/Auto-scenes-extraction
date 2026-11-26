"""固定五档的单帧美学量表。权重用于当前的短片选帧任务。"""
from math import isclose
from statistics import mean


# 每项依次对应 0、25、50、75、100。两次相反的选项顺序共用同一份视觉特征。
SCALES = {
    "景别取景": (0.10, "比较画框边缘、留白和画内线条。它们怎样组织视线？", (
        "裁切或遮挡破坏重点，视线找不到落点。",
        "有可辨重点，但突兀的边界或抢眼的杂物持续分散注意。",
        "重点清楚，取景完整；其他元素对视线路径作用很小。",
        "一处明显的留白、线条或框景把视线带到重点。",
        "多个区域构成清楚的观看路径，落点与画框边缘也有呼应。")),
    "光影": (0.16, "看光落在哪里、阴影留下什么形状。光影给画面增加了什么？", (
        "主要形状消失在杂乱的明暗之中。",
        "亮暗能勉强交代形状，但炫光或阴影遮掉关键关系。",
        "光线交代清楚事物，主要作用是照亮。",
        "一处明暗交界、逆光或柔光塑造了轮廓、层次或落点。",
        "亮区与暗区各有作用，既塑造空间或形体，又共同引导视线。")),
    "色调": (0.11, "看整图色块的分布和过渡，黑白画看灰阶。饱和与黑白本身都不决定档位。", (
        "色块或灰阶混杂，视觉重心难以辨认。",
        "主要色调能辨认，局部突兀色块或灰阶破坏了它。",
        "配色或灰阶统一，主要作用是交代各区域。",
        "有明确的色彩呼应或灰阶对照，帮助突出重点。",
        "主调、过渡与强调色或灰阶各司其职，形成有辨识度的整幅关系。")),
    "主体表现": (0.12, "", ()),  # 问题和档位按主体、景别选择。
    "人物表情情绪": (0.22, "只评人物可见脸部表情传达情绪的清晰度、自然度和细节。"
                             "悲伤、愤怒、平静等任何情绪都可得高分；不要把妆容或漂亮程度算作表情分，也不要以情绪强烈程度代替表达质量。", (
        "表情线索互相矛盾，难以辨认人物表达的情绪。",
        "五官可见，但表情线索弱或含混，很难判断表达的情绪。",
        "能看出大致情绪方向，主要依靠一处明显线索，表达较常见。",
        "眼神、眉毛或嘴角等多处线索相互吻合，传达明确且自然的情绪。",
        "表情细微而有层次，五官线索共同呈现可信、具体的情绪状态，不靠夸张强度取胜。")),
    "整体画面氛围": (0.24, "只评画面构成营造的整体氛围，不评人物脸上的情绪是否清楚。"
                             "观察光色、构图、空间、主体与环境的关系是否共同形成特定感受；悲伤、紧张、宁静、诡谲等任何氛围都可得高分。"
                             "不要依赖作品背景、角色身份或熟悉题材来推断；鲜艳、精致或清晰本身不等于氛围强。", (
        "画面线索互相冲突或过弱，难以形成稳定的整体感受。",
        "大致能辨认题材或情绪方向，但缺少画面关系支撑这种感受。",
        "一处可见线索给出基本基调，整体氛围较常见或单薄。",
        "至少两类画面线索相互支持，形成明确且具体的整体氛围。",
        "主体、光色、空间或环境彼此呼应或形成有意反差，构成鲜明、连贯且有层次的情绪体验。")),
    "技术呈现": (0.05, "按当前景别看关键边界。剪影、浅景深、柔光、颗粒只在妨碍阅读时扣分。"
                         "技术质量不是美学价值本身，只记录它是否妨碍氛围和主体阅读。", (
        "明显的损坏、重影或模糊使主要内容无法辨认。",
        "局部锯齿、糊块或高光丢失妨碍关键部分的阅读。",
        "主要信息可读，个别边缘、纹理或层次含混。",
        "重要边界和层次清楚，软硬过渡与画面风格相符。",
        "在亮暗、细密或柔化等难处理区域，关键边界与微妙层次也得到保留。")),
}

SUBJECT_SCALES = {
    "人物近处": ("脸、身体局部或可见姿势的造型如何？", (
        "关键轮廓被破坏，人物形态难读。", "看得出人物，但主要姿态或局部被遮乱。",
        "人物或局部清楚，朝向和姿势主要起说明作用。",
        "眼神、姿势或轮廓中有一处鲜明的表达。", "姿势、朝向或手势与画框内其他形状呼应，构成独特瞬间。")),
    "人物远处": ("看人物相对周围空间的尺度、位置与朝向。", (
        "人物难以从周围形状中辨认。", "能找到人物，周围形状明显干扰其轮廓。",
        "人物位置清楚，和环境主要是并列呈现。", "人物的位置、大小或朝向与空间有明显对照。",
        "人物与前后景形成明确的尺度或方向关系，这一安排强化了画面的含义。")),
    "物体": ("看可见材质、外形或结构，图形只看线条与色块。", (
        "主要物体的外形难以辨认。", "物体可辨，但关键特征被遮挡或粘连。",
        "能看出是什么，取景主要展示普通外观。", "材质、轮廓或结构中的一项成为明确的视觉看点。",
        "多个形状或材质关系协调，揭示了这个对象特有的造型。")),
    "风景": ("看景物的尺度、分布、留白及前后关系。平面风格看形块关系。", (
        "主要景物混在一起，难以读出布局。", "能辨认场所，主要区域互相干扰。",
        "地点和布局清楚，景物主要是并排呈现。", "景物间的疏密、尺度或遮挡形成明确的节奏。",
        "主景与周围多处景物构成有特点的空间秩序，近看与整体阅读都成立。")),
}


EVIDENCE = {
    "景别取景": ("构图中有没有一处具体关系，让眼睛从一个区域自然移向另一处重要位置？",
                 "线条、留白、遮挡或重复形状，在两处之间形成可追踪的路径。",
                 "主要靠认出画面里的对象，找不到清楚的第二处落点或路径。"),
    "光影": ("若把各处改为一样亮，会失去什么可见关系？",
             "有具体的明暗交界组织了视线、形体或空间，改成平光会破坏这种关系。",
             "亮暗主要让内容可见，换成平光对形体或空间关系影响有限。"),
    "色调": ("比较相隔较远的两个区域。色块或灰阶之间有没有明确的呼应或对照？",
             "能指认有联系的两个区域，色彩或灰阶的安排突出了其中的重点。",
             "颜色或灰阶总体协调，但找不到这种有明确作用的区域关系。"),
    "人物表情情绪": ("只看可辨的脸部线条，眼睛、眉毛或嘴角是否共同表达了具体情绪？",
                 "能指出至少两处互相吻合的五官线索，传达清楚、自然或细腻的情绪。",
                 "脸部虽可见，但情绪主要靠猜测，或缺少清楚一致的表情线索。"),
    "整体画面氛围": ("不看人物表情，也不依赖作品背景或熟悉题材，画面本身是否通过光色、构图、空间或主体关系营造出具体氛围？",
                 "不同画面线索彼此支持或形成有意反差，共同营造明确、连贯的氛围。",
                 "印象主要来自题材或单一视觉线索，其他画面关系对氛围作用有限。"),
    "技术呈现": ("找最难处理的可见部位：亮暗交界、相邻细线、纹理或柔化边缘。其处理是否有明显优点？",
                 "有可指出的部位，既保留必要区别又没有明显的断层、糊块或重影。",
                 "整体可以读懂，但找不到这样的处理实例，或关键处确实有损失。"),
}

FORM_EVIDENCE = {
    "人物近处": ("除能认出人物之外，可见姿态或表情是否与画框内另一元素构成具体关系？",
                "目光、手势或身体线条在画中有可指出的呼应或对象。",
                "以常见姿态展示人物，造型的作用基本到此为止。"),
    "人物远处": ("想象去掉人物。场景会失去什么？",
                "空间的尺度、方向或孤立感会实质改变，人物位置与场景共同表达了东西。",
                "场所的观感基本相同，人物主要只是出现于其中。"),
    "物体": ("这件物体除了容易辨认，画面是否揭示了形态、材质或用途的一处独特关系？",
             "一个部件、反光或外形与周围发生具体联系，使物体有独到的看点。",
             "主要是普通地展示外形，作用集中在辨认物体。"),
    "风景": ("相邻或前后景之间有没有形成独特的空间、疏密或形块关系？",
             "能具体指出不同景物如何彼此框住、呼应或对照，共同构成布局。",
             "画面交代了地点和景物，但它们之间的组织较常见。"),
}


def paired_order_scores(engine, features, record, questions):
    """还原选项顺序后取两次预测的均值。"""
    sums = {q["id"]: [0.0] * len(q["options"]) for q in questions}
    for reverse in (False, True):
        batch = [{**q, "options": list(reversed(q["options"])) if reverse else q["options"]} for q in questions]
        answers = engine.model.predict_features(engine.processor, features, record, batch)
        for q, answer in zip(batch, answers, strict=True):
            probs = answer["probabilities"]
            if (answer["id"] != q["id"] or len(probs) != len(q["options"]) or
                    not all(0 <= p <= 1 for p in probs) or not isclose(sum(probs), 1, abs_tol=1e-5)):
                raise ValueError(f"无效的选项概率：{q['id']}")
            for i, p in enumerate(reversed(probs) if reverse else probs):
                sums[q["id"]][i] += p / 2
    return sums


MOOD = "整体画面氛围"
FACE = "人物表情情绪"

# 选项按场景承担的表达作用排序。题目不显示分值，也不使用同批排名。
MOOD_QUESTIONS = {
    "表现目的": ("哪一种更接近这张图的主要表现目的？只看可见的画面，不猜作者。", (
        "让人看清一个角色、物件或地点。", "让人留意动作、造型或环境细节。",
        "让人感到一个场所的特定气息。", "用空间与光色传递一种鲜明处境。")),
    "视觉重心": ("哪句更准确地描述画面当前最突出的东西？不考虑故事，也不考虑对象本身讨不讨人喜欢。", (
        "主体的可辨外观。", "局部的漂亮光色。", "画面各处共同传递的气息。")),
    "焦点与周围": ("把视线从最抢眼的局部移向周围，哪一种情况最贴近原画？局部也可以是一块光或一个形状。", (
        "周围没有留下可辨的情绪线索。", "周围主要交代对象在哪里。",
        "周围的明暗或间距改变了观看重点的感受。",
        "重点与周围形成呼应或张力，单看一处便失去原图意味。")),
}


def assess(engine, features, record, subject, fields, *, evidence=False, scene_cues=False,
           face_visible=None, face_or_mood=True):
    profile = subject
    if subject == "人":
        profile = "人物远处" if fields["景别"] in ("全景", "远景") else "人物近处"
    details = [fields.get(k) for k in ("姿势", "动作", "物体细类", "场景细类")]
    details = [v for v in details if v and v not in ("无法判断", "无明确动作", "其他")]
    active_dimensions = list(SCALES)
    # Facial emotion is not a missing-data penalty: score it only when a person
    # is present and the face exposes enough cues to classify its emotion.
    if face_visible is None:
        face_visible = fields.get("人物情绪") not in (None, "无法判断")
    if subject != "人" or not face_visible:
        active_dimensions.remove(FACE)
    elif scene_cues and face_or_mood:
        active_dimensions.remove(MOOD)
    questions = []
    for name, (_, query, levels) in SCALES.items():
        if name not in active_dimensions:
            continue
        if scene_cues and name == MOOD:
            questions.extend({"id": key, "task": "choice", "query": prompt, "options": list(options)}
                             for key, (prompt, options) in MOOD_QUESTIONS.items())
            continue
        if name == "主体表现":
            query, levels = SUBJECT_SCALES[profile]
            if details:
                query += "参考可见的" + "、".join(details) + "。"
        elif name == FACE and not scene_cues:
            query += f"上一步面部情绪分类为「{fields['人物情绪']}」；请重新依据五官线索判断，不能只因该标签而加分。"
        questions.append({"id": name, "task": "choice", "query": f"按{fields['景别']}观察。{query}选择与画面最相符的一档。",
                          "options": list(levels)})

    sums = paired_order_scores(engine, features, record, questions)
    question_scores = {key: 100 * sum(i * p for i, p in enumerate(probs)) / (len(probs) - 1)
                       for key, probs in sums.items()}
    scores = {key: value for key, value in question_scores.items() if key in active_dimensions}
    if evidence:
        items = {k: v for k, v in EVIDENCE.items() if k in scores}
        items["主体表现"] = FORM_EVIDENCE[profile]
        checks = [{"id": name, "task": "choice", "query": text, "options": [basic, exceptional]}
                  for name, (text, exceptional, basic) in items.items()]
        gates = paired_order_scores(engine, features, record, checks)
        scores = {k: .60 * v + .40 * 100 * gates[k][1] for k, v in scores.items()}
    weights = {name: SCALES[name][0] for name in active_dimensions}
    if scene_cues:
        weights = dict(景别取景=.08, 光影=.10, 色调=.07, 主体表现=.09, 技术呈现=.06)
        if MOOD in active_dimensions:
            scores[MOOD] = mean(question_scores[key] for key in MOOD_QUESTIONS)
        if FACE in scores and MOOD in scores:  # V9 双项计分。
            weights[MOOD], weights[FACE] = .40, .20
        else:
            weights[FACE if FACE in scores else MOOD] = .60
    score = sum(scores[name] * weight for name, weight in weights.items()) / sum(weights.values())
    by_name = {q["id"]: q for q in questions}
    result = {"subject": subject, "aesthetic_score": round(score, 2),
              "细分": {**fields, "细项评分": {
                  name: round(scores[name], 2) if name in scores else None for name in SCALES}}}
    if scene_cues:
        if MOOD not in scores:
            index = max(range(len(sums[FACE])), key=sums[FACE].__getitem__)
            result["细分"]["表情判断"] = by_name[FACE]["options"][index]
            return result
        key = min(MOOD_QUESTIONS, key=lambda k: abs(question_scores[k] - scores[MOOD]))
        index = max(range(len(sums[key])), key=sums[key].__getitem__)
        result["细分"]["氛围判断"] = by_name[key]["options"][index]
        return result
    observations = {}
    for label, name in (("relative_strength", max(scores, key=scores.get)),
                        ("relative_weakness", min(scores, key=scores.get))):
        if evidence:
            description = items[name][1] if gates[name][1] > .5 else items[name][2]
        else:
            index = max(range(len(sums[name])), key=sums[name].__getitem__)
            description = by_name[name]["options"][index]
        observations[label] = {"observation": f"{name}：{description}"}
    result["review"] = observations
    return result
