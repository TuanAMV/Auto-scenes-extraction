"""单图景别、姿态与细类，以及据此选用的美学问题。叶子词参考 Auto-scenes-extraction/logic_keywords.json。"""

UNKNOWN = "无法判断"
OTHER = "其他"
NO_ACTION = "无明确动作"

# 每行包含该类的评分观察点和代表关键词。姿势、持物及自然地貌另补常见单图用词。
OBJECTS = {
    "动物或生物": ("外形、翅羽或肢体姿态", "动物_猫 动物_鸟 动物_鱼 动物_狼 动物_老虎 兔子 蝴蝶 昆虫 蛇 水母 怪物"),
    "植物或果实": ("叶形、花瓣或枝条的层次", "树 树叶 鲜花 蒲公英 荆棘 水果 苹果 橘子"),
    "天体或光象": ("明暗边界、光晕和周围暗部", "月亮 太阳 星星 星球 日食 闪电 极光 烟花 彩色光"),
    "建筑或构件": ("边线、窗格和结构比例", "建筑 窗子 门 风车 电塔 电线杆 电梯 红绿灯 神像 石碑"),
    "设备或机械": ("外壳、操作面和零件的关系", "手机 屏幕 电脑 相机 电视机 耳机 音箱 仪表盘 齿轮 机械 机甲 仪器"),
    "交通工具": ("车身或机身的轮廓与朝向", "汽车 摩托车 列车 飞船 卫星 潜艇 直升机 飞碟"),
    "器械或武器": ("器物的形状、边缘和握持部位", "武器 枪械 盾牌 弓箭 刀 剑 斧头 镰刀 锯子"),
    "生活用品或饰物": ("材质、表面细节和功能轮廓", "宝石 戒指 钟表 玩偶 面具 杯子 鞋子 伞 药瓶 玻璃 蜡烛 钥匙"),
    "纸品图形或文娱": ("文字、纹样或器形的辨识度", "卡牌 书 纸张 报纸 海报 地图 符号 棋子 吉他 钢琴 麦克风 唱片机"),
    "自然物质": ("材质边缘、纹理和透光变化", "水滴 涟漪 岩浆 烟 泡泡 露珠 流体 火焰 石头"),
}
SCENES = {
    "林野与地貌": ("地形轮廓、植被疏密与层次", "森林 荒野 麦田 火山 山地 草原 沙漠"),
    "水岸与海域": ("水面、岸线或远近水域的层次", "海洋 湖泊 沙滩 河流 瀑布"),
    "天空与太空": ("云层、天体、留白与明暗过渡", "天空 云朵 星空 太空"),
    "城镇与街区": ("建筑群、路面与街道的空间关系", "城市 街道 巷子 村子 轨道 停机坪"),
    "室内与建筑空间": ("墙面、开口、陈设和空间纵深", "室内 房间 走廊 医院 学校 实验室 工厂 教堂 电影院 水族馆"),
    "幻想与特殊环境": ("异常空间中的尺度线索和视觉中心", "赛博空间 赛博城市 废墟 末日 倒悬"),
    "平面或简单背景": ("色块分区、留白和边界", "简单背景 纯色背景 抽象图形"),
}
ACTIONS = {
    "头部视线或表情": ("眼神、头部朝向或表情", "仰望 回头 相视 瞭望 笑脸 哭泣 呐喊 闭眼 低头"),
    "手势或持物": ("手指轮廓、手势和物件之间的接触", "伸手 伸出手掌 握拳 握持 触摸 弹指 嘘手势 鼓掌 投掷 翻页"),
    "步态或全身动作": ("重心、肢体线条和运动方向", "行走 奔跑 跳跃 转身 跳舞 下跪 躺下 摔倒 下潜"),
    "人物间接触": ("两人之间的距离、手势和相互朝向", "拥抱 搂抱 牵手 握手 碰拳 背对背 亲吻"),
    "生活或表演动作": ("动作涉及的器物、接触点和姿势", "进食 喝水 睡觉 祈祷 唱歌 演奏 画 化妆 干杯"),
    "对抗动作": ("动作方向、发力姿态与目标之间的关系", "战斗 击打 劈砍 开枪 躲避 格挡"),
}
POSES = {
    "站姿": "站姿，身体主要靠伸直的腿支撑", "坐姿": "坐姿，臀部由座面或地面支撑",
    "蹲姿": "蹲姿，双腿明显屈曲，重心低", "跪姿": "跪姿，膝部着地或抵住支撑面",
    "卧姿": "卧姿，身体躺在支撑面上", "腾空": "腾空，身体明显离开支撑面",
    "俯身": "俯身，躯干明显向前弯", UNKNOWN: "只见脸或局部、身体被遮挡，无法判断支撑姿势",
}
SHOT_LABELS = ("大特写", "特写", "近景", "中景", "全景", "远景")
# 三个主体共用标签，采用各自适用的尺度参照。
SHOTS = {
    "人": ("眼、唇、手指等很小的身体局部占据画面", "脸或一只手占画面大部",
          "头肩到胸部入画", "腰部或膝部以上入画，能看清上身姿态",
          "基本完整的人物从头到脚入画", "环境占画面大部，人物很小"),
    "物体": ("单个纹理、表面痕迹或微小零件充满画面", "关键部件或结构局部占主导",
            "物体占画面大部，基本可辨整体", "物体和紧邻它的道具或支撑面一起入画",
            "物体全貌及所在空间均清楚", "大范围环境中，作为视觉重点的物体很小"),
    "风景": ("单一景物的微小纹理铺满画面", "只展示一小片枝叶、石壁或建筑细节",
            "取景集中在一处局部景物", "几处相邻景物共同构成一段局部空间",
            "一处房间、街角、林间空地等场所的全貌", "广阔地形、天际线或纵深很长的空间"),
}


def _question(key, text, options):
    return {"id": key, "task": "choice", "query": text, "options": list(options)}


def _ask(engine, features, record, *questions):
    answers = engine.model.predict_features(engine.processor, features, record, list(questions))
    values = []
    for q, a in zip(questions, answers, strict=True):
        if a["id"] != q["id"] or a["answer"] not in q["options"]:
            raise ValueError(f"分类结果与问题不一致：{q['id']}")
        values.append(a["answer"])
    return values


def _groups(groups, fallback):
    return {key: f"{key}（如{'、'.join(words.split()[:3])}）" for key, (_, words) in groups.items()} | {fallback: fallback}


def describe(engine, features, record, subject, emotion_question, *, include_face_visibility=False):
    """复用同一帧的特征；只展开被选中的分支。"""
    shot_options = [f"{label}：{desc}" for label, desc in zip(SHOT_LABELS, SHOTS[subject], strict=True)]
    shot = _question("shot", "按本图主要视觉对象的取景范围选择景别。", shot_options)
    groups = {"人": ACTIONS, "物体": OBJECTS, "风景": SCENES}[subject]
    fallback = NO_ACTION if subject == "人" else OTHER
    options = _groups(groups, fallback)
    instructions = {
        "人": "画中主角可直接辨认的主要动作是哪一类？仅有中性站坐、普通注视或证据不足，选无明确动作。",
        "物体": "占据视觉焦点的这个独立对象属于哪一类？手拿的物品看物品本身。无匹配选其他。",
        "风景": "画面表现的主要空间属于哪一类？优先根据可见建筑、地貌等选具体空间。无匹配选其他。",
    }
    questions = [shot, _question("family", instructions[subject], options.values())]
    if subject == "人":
        questions.extend([_question("pose", "主角的身体如何受到支撑？看不清身体时请选择无法判断。", POSES.values()),
                          emotion_question])
    elif subject == "风景":
        questions.append(_question("appearance", "这处环境最清楚的天气或光照特征是什么？以画面中可见的迹象为准。",
                                   ["雨", "雪", "雾", "夕照", "夜色", "霓虹", "晴朗", "无明显特征"]))
    framing, family_answer, *extra = _ask(engine, features, record, *questions)
    scale = SHOT_LABELS[shot_options.index(framing)]
    family = next(key for key, value in options.items() if value == family_answer)
    focus = {"人": "可见的头部朝向、身体轮廓或手势", "物体": "主体可见的材质和轮廓",
             "风景": "可见景物的大小、遮挡和留白"}[subject]
    detail = fallback
    if family in groups:
        focus, words = groups[family]
        words = words.split()
        detail, = _ask(engine, features, record, _question("keyword",
            f"在「{family}」中，哪一个词最符合本图可见的主要对象或动作？选最具体且有证据的词。"
            f"有别的表现选「其他」，线索不足选「{UNKNOWN}」。", [*words, OTHER, UNKNOWN]))
    if subject != "人" and detail not in (OTHER, UNKNOWN, family):
        evidence, = _ask(engine, features, record, _question("detail_evidence",
            f"图中有没有能把「{detail}」与同类{family}区别开的可见标志？仅凭常见情境猜测时选第二项。",
            ["可见能辨别该细类的标志", "只能辨认大类或标志不清"]))
        if evidence == "只能辨认大类或标志不清":
            detail = UNKNOWN
    fields = {"景别": scale}
    if subject == "人":
        fields["姿势"] = next(k for k, v in POSES.items() if v == extra[0])
        fields["动作"] = family if detail == OTHER else detail
        cues, = _ask(engine, features, record, _question("face_visibility",
            "主角脸部有哪些能看清的细节？以眼睛、眉毛或嘴角的具体线条为依据。",
            ["纯色剪影、背影或遮挡，看不清五官", "可以分辨眼眉或嘴角的表情线条"]))
        visible = cues == "可以分辨眼眉或嘴角的表情线条"
        fields["人物情绪"] = extra[1] if visible else UNKNOWN
        if include_face_visibility:
            fields["面部可辨"] = visible
    elif subject == "物体":
        fields.update(物体类别=family, 物体细类=detail)
    else:
        fields.update(风景类别=family, 场景细类=detail, 环境特征=extra[0])
    return fields, focus


def _check(key, question, levels):
    return _question(key, question, levels)


def environmental_portrait_checks(checks, subject, fields):
    """按亮暗对观看路径的作用评分。全景和远景按其观看尺度判断清晰度与人物造型。"""
    if subject != "人":
        return checks
    replacements = {
        "光色重心": _check("光色重心", "先看最亮或对比最强的区域，再看视线接下来停在哪里。这样的明暗安排对整幅画面有什么作用？",
                      ["亮暗落点相互争抢，视线被引到与画面无关的地方。",
                       "主要区域能分辨，但明暗对视线走向的帮助有限。",
                       "亮暗关系能引导视线或分出空间层次，亮处与暗处共同突出画面重点。"]),
    }
    if fields["景别"] in ("全景", "远景"):
        replacements.update({
            "焦点边缘": _check("焦点边缘", "在整幅画面的观看尺度上，人物外轮廓及其所在空间的主要边界是否容易分开？",
                           ["主要形状与背景混在一起，空间关系也难读。",
                            "主要形状可辨，部分关键轮廓略显含混。",
                            "人物轮廓与主要空间层次易于分辨，柔光或暗部仍保留所需边界。"]),
            "动作可读性": _check("动作可读性", "看人物与地面、前后景之间的轮廓关系。人物在空间里的位置是否读得清？",
                           ["人物与环境的边界混杂，很难辨认其位置。",
                            "能辨认人物位置，局部形状与背景混在一起。",
                            "人物的位置和整体形状一眼可读。"]),
            "主体造型": _check("主体造型", "从整幅画面看，人物的大小、位置或朝向与空间形成了怎样的关系？",
                          ["人物与周围形状杂乱重叠，难以看出两者的关系。",
                           "人物和环境均可辨，两者的安排平常。",
                           "人物的尺度、位置或朝向与周围空间形成鲜明的对照或呼应。"]),
            "环境呼应": _check("环境呼应", "不猜测人物内心。只看明暗、色调、人物位置和空间，画面的情绪氛围是否成立？",
                          ["视觉线索互相冲突或太弱，很难感到明确基调。",
                           "能感到大致基调，主要由单一线索支撑。",
                           "多处可见线索配合，整幅画面呈现明确而连贯的氛围。"]),
        })
    return [(dimension, replacements.get(q["id"], q)) for dimension, q in checks]


def adapt_checks(checks, subject, fields, focus):
    """替换五维量表里的适用子题，维度数与权重保持一致。"""
    scale = fields["景别"]
    if scale in ("大特写", "特写"):
        target = "特写局部的关键边界、裁切位置和周围留白"
    elif scale in ("近景", "中景"):
        target = "重点形状、朝向空间和周围干扰的关系"
    else:
        target = "主要轮廓在整体空间中的位置、层次和留白"
    framing = _check("景别取景", f"按{scale}来评价{target}。以画面选择展示的范围为准，取景效果如何？",
                     ["关键轮廓被画边截乱或混入杂物。", "重点可辨，取景与留白大致合适。", "边缘、留白与重点配合，所选范围的视觉重心明确。"])
    if subject == "人":
        cues = [fields[k] for k in ("姿势", "动作") if fields[k] not in (UNKNOWN, NO_ACTION, OTHER)]
        description = "、".join(cues) or "可见的人物局部"
        question = f"按{scale}观察{description}。只看可见部分，{focus}在造型上是否清楚、有表现力？"
        levels = ["可见轮廓相互遮挡，重心或朝向难辨。", "人物可见姿态或局部形状可辨。", "人物的线条、朝向或手势形成鲜明而连贯的造型。"]
    elif subject == "物体":
        description = fields["物体细类"] if fields["物体细类"] not in (UNKNOWN, OTHER) else fields["物体类别"]
        question = f"观察{description}在此{scale}中实际呈现的{focus}。哪档最符合它的造型？"
        levels = ["关键特征粘连或被掩盖，造型难认。", "外形和主要特征可以辨认。", "轮廓、光色或纹理突出了这个对象独有的特征。"]
    else:
        description = fields["场景细类"] if fields["场景细类"] not in (UNKNOWN, OTHER) else fields["风景类别"]
        question = f"观察{description}在此{scale}下的{focus}。可见形状怎样组织成这处场景？"
        levels = ["主要景物相互干扰，空间或图案难读。", "能辨认主要景物及其布局。", "景物的疏密、尺度或留白形成清楚而有特点的布局。"]
    form = _check("主体造型", question, levels)
    result = []
    for dim, q in checks:
        if q["id"] in ("取景边缘", "裁切留白", "空间层次"):
            q = framing
        elif q["id"] in ("轮廓造型", "造型看点", "形状节奏"):
            q = form
        elif subject == "人" and q["id"] == "动作可读性" and fields["动作"] not in (UNKNOWN, NO_ACTION):
            q = _check("动作可读性", f"看{scale}里与「{fields['动作']}」有关的脸、手或肢体。单凭它们能否读出这个动作？",
                       ["关键接触点或动作方向被遮住。", "能读出动作，但部分线索重叠。", "动作方向、接触点或目光清楚，一眼能读出。"])
        elif subject == "物体" and q["id"] == "结构显现":
            q = _check("结构显现", f"看{description}的{focus}。亮处和暗处分别保留了多少关键形状？",
                       ["高光或暗部淹没了关键特征。", "基本形状能辨认，局部结构不清楚。", "明暗过渡使材质或关键结构清晰可辨。"])
        elif subject == "风景" and q["id"] == "色调分层":
            weather = fields["环境特征"]
            lighting = "画面的光线" if weather == "无明显特征" else f"画面的{weather}条件"
            q = _check("色调分层", f"在{lighting}下，比较{description}中不同区域的亮暗或色调。主要景物是否突出？",
                       ["大块明暗或颜色互相干扰，主景难辨。", "主要区域可以区分，但视觉重点一般。", "光色过渡清楚，主景突出且空间层次可读。"])
        elif subject == "人" and q["id"] == "人物线索" and (scale in ("全景", "远景") or fields["人物情绪"] == UNKNOWN):
            q = _check("姿态表达", "按当前景别只看能见的剪影、人与环境的距离及朝向，它们表达的基调有多清楚？",
                       ["可见线索零散，基调难读。", "姿态或空间给出大致基调。", "轮廓、朝向和环境关系共同传达清楚的基调。"])
        result.append((dim, q))
    return result
