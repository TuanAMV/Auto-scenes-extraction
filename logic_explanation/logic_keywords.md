
该方案旨在将您的原始视频素材关键词文档转化为结构化、逻辑严密且视觉精准的英文提示词（Prompt），用于从视频帧中批量检索特定画面。

1. 核心策略：视觉原型与纯净逻辑
我们摒弃了“大海捞针”式的关键词堆砌，转变为**“视觉原型 + 物理逻辑 + 介词流”**的生成模式。

视觉原型化 (Visual Prototypes)：

归类泛化：将具体的社会身份（如：警察、黑客）统一归类为视觉核心词（如：man），利用 CLIP 的泛化能力覆盖所有男性角色。

武器独立化：将“武器”从普通物体中剥离，建立独立的 Weapon 大类。将具体型号（步枪、冷兵器）映射为视觉特征明确的单词（gun, sword, futuristic rifle），以匹配战斗或战损场景。

去时序化 (De-temporalization)：剔除 CLIP 无法理解的时间/剪辑类关键词（如“时间变化”、“镜头拉伸”、“转场”）的干扰。

保留状态：仅保留具有强视觉特征的瞬间状态（如“悬浮”、“破碎”、“燃烧”）。

2. 逻辑架构：多维物理约束
为了防止生成违反物理常识的废片（如“汽车在哭泣”或“手枪在进食”），我们将原有的二元逻辑升级为五大类逻辑判定：

生物/器官 (Biotic/Organ)：

定义：人、动物、怪物、巨大的眼睛/心脏。

逻辑：全能型。可拥有情绪行为（哭泣/尖叫）、姿态（下跪/祈祷）、运动及物理状态。

载具 (Vehicle)：

定义：汽车、飞船、机甲。

逻辑：仅限 运动状态（冲撞/飞行）+ 物理状态（爆炸/生锈）。绝无生物行为。

武器 (Weapon)：

定义：枪械、刀剑、火炮。

逻辑：高动态物体。混合了 运动状态（发射/悬浮）+ 物理状态（发光/破碎）。严禁生物情绪（武器不会哭，也不会笑）。

物体/环境 (Object)：

定义：建筑、树木、静物。

逻辑：被动状态。仅限 物理状态（燃烧/倒塌）或 被动位移（漂浮/下坠）。

3. 提示词结构：纯介词逻辑流
为了避免额外的动词（如 depicting, capturing）被 CLIP 模型误识别为画面内容，我们采用**纯介词（Prepositions）**来构建逻辑关系。

生成公式：

Plaintext
A {mood} {lens} of a {subject} {action} in {scene}
结构解析：

A {mood} {lens}：定义画面的基调与视角（如：A horror close-up / 一个恐怖特写）。

of：[内容指向]。告诉模型接下来的词是画面中的主体（如：of a monster screaming / 关于一个正在尖叫的怪物）。

in：[空间指向]。告诉模型主体所处的环境（如：in forest / 在森林里）。

最终输出示例（无标点单句）：

A cyberpunk close-up of a futuristic rifle glowing in battlefield

A cinematic high angle of a spaceship exploding in space

A horror wide shot of a monster screaming in forest