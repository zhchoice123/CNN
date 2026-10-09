# 代码练习的写法

每篇练习 = **一份能直接运行的完整答案脚本** + 注释标记。构建器从它生成四级练习：

| 级别 | 读者看到 | 怎么判 |
|---|---|---|
| L1 选择 | 代码里 `mask` 的位置挖空，给选项 | 浏览器里比对，离线可用 |
| L2 填空 | 同上，自己输入 | 去空格、统一引号和数字写法后比对，`@accept` 可加等价写法 |
| L3 写整块 | 挖空的**整段代码**换成 `# ✏️ 在这里写代码` | 在本机运行，按步骤执行 `@check` |
| L4 默写 | 只剩步骤标题和 `@todo` / 提示 | 同上 |

## 标记

```python
"""
@title: 标题                       ← 必填
@level: 入门 / 进阶
@dataset: mnist                    ← 说明用
@chapter: ch06                     ← 关联学习站章节（章节页底部会出现入口）
@minutes: 30
@goal: 这一篇练什么                 ← 必填
@task: 读者要完成的任务             ← 必填，显示在 L3/L4 模板开头
@names: device, model, optimizer    ← 评测要用到的变量名，L3/L4 模板会提示读者使用这些名字
"""
# @step data 读取数据集               ← 新步骤：id + 标题。必须顶格
# @explain                           ← 讲解（支持 - 列表、1. 列表、> 提示、`代码`、**粗体**、$公式$、```代码块```）
# …
# @end
# @todo: 显示给读者的提示注释（L3/L4 里变成 # TODO:）
# @blank loader mask=shuffle=True    ← 挖空：id + mask（必须出现在下面的代码里，出现几次就挖几次）
# @hint: L1/L2 的提示                 ← 必填
# @option: shuffle=False             ← 干扰项（至少 1 个；正确答案 = mask，会自动加入）
# @accept: shuffle = True            ← L2 的等价写法（可多条）
# @why: 为什么是这个答案               ← 必填
# @bad: train_loader = DataLoader(train_set, batch_size=128)   ← 错误样例：验收时必须在本步被检出
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
# @endblank
# @check: assert …, "检查不通过时给读者看的提示"     ← 评测时在本步末尾执行（单行 Python，可用 ; 连接）
# @fast: train_set = Subset(train_set, range(6000))  ← 只在评测时执行，用来缩小数据或轮数
```

## 规则（构建器会强制检查）

1. **步骤边界必须顶格**：评测把读者的代码按“步骤标题行”切开，在每步末尾注入 `@check` 和 `@fast`。所以类定义、循环不能跨步骤。
2. **`@fast` 后面不能再有代码**：它只在步骤末尾生效。要先缩小数据集再建 DataLoader，就把 DataLoader 拆到下一步；要缩短轮数，就把 `EPOCHS = …` 放在训练循环之前的一个步骤里。
3. 步骤 id 和挖空 id 在同一篇里不能重复。
4. 检查要写成**行为检查**（形状、损失是否下降、准确率阈值、两次预测是否一致），不要比对字符串，这样等价写法都能通过。
5. 阈值要留余量：MPS 和 CUDA 的数值有细微差异；写错的代码通常只有 10% 左右的准确率，阈值设在快速模式实测值以下 15–20 个百分点即可。

## 构建与验收

```bash
python3 study_site/tools/build_exercises.py     # 生成 assets/js/exercises-data.js（只做格式检查，很快）
python3 lab/verify_exercises.py                 # 每台机器都要跑：答案全部通过、错误样例被检出、单次评测 ≤ 30 s
python3 study_site/tools/build_codelib.py       # 把答案收录进代码库（可选）
```
