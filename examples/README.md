# 示例：两套真实数据上的端到端 demo

这里把七步流水线完整跑两遍：一遍用 **ArUco（ChArUco 板）**数据，一遍用
**YCBV** 数据。两次都从第 04 步开始，因为这两套数据在磁盘上已经带了
第 01–03 步的产物（COLMAP 稀疏重建、训练好的 3DGS、逐位姿渲染图）。

| demo | 数据 | 渲染帧 | 控制点 | 目录 |
|---|---|---|---|---|
| ArUco | `charuco/train/0000` | 946 | 15 个板面控制点 | [`charuco/`](charuco/) |
| YCBV | `ycbv_little/data/train/seq00_room_20261012` | 1000 | 19 个物体关键点 | [`ycbv/`](ycbv/) |

## 唯一被“模拟”的一步

第 04/06 步要求用 GUI 在渲染图上点控制点。demo 要无人值守运行，所以
`make_clicks.py` 把数据集**已经有的渲染域标签**原样写成标注 JSON：

* ArUco：`3dgs/renders/labels/*.json`（在渲染图上直接检测 ChArUco 标志得到，
  15 个亚像素控制点），
* YCBV：`labels_render/points_states.csv` 的 `state=visible` 行（光流从真实
  图像搬过来的渲染域标签）。

写出的 JSON 与 `renderlabel annotate` 的存档格式完全一致，所以第 05/06/07 步
以及评测看到的都是**流水线本体**。这段替代只发生在“点击”这一步，且脚本自己
在输出里写明 `simulated clicks ...`。

## 运行

```bash
cd /path/to/render-label-pipeline

# ArUco
SCENE=/path/to/charuco/train/0000 bash examples/charuco/run_demo.sh

# YCBV
SCENE=/path/to/ycbv_little/data/train/seq00_room_20261012 \
    bash examples/ycbv/run_demo.sh
```

可选环境变量：`RENDERS`、`LABELS`、`GT_POINTS`（仅 YCBV）、`WORK`、`PY`
（默认 `python3`）、`THETA_EFF`、`GUARD`、`MARGIN`、`MIN_VIEWS`（默认 2）、
`TRIALS`。
脚本只需要 `numpy` / `opencv-python`；`pycolmap` 只用于 `check_stages.py` 的
第 01–02 步核验。

每次运行会在 `examples/<demo>/work/` 下留下：

```
selection.json      种子帧、Theta_span、N_sat、选中的帧
points3d_seed.json  种子交会出的粗三维点（第 06 步的引导圈）
seed.json           种子帧的“点击”
selected.json       选中帧的“点击”
labels/             第 07 步输出：每张渲染图一个 labelme JSON + points3d.csv + summary.json
eval.json           留出集 2D 误差、随机基线、与真值三维的对比
summary.md          summarize.py 从上面几个文件生成的汇总表
```

## 指标怎么读

* **annotated-view residual**：交会用的那些帧上的重投影残差。它只说明
  “点击和位姿自洽”，不是精度，因为这些都是参与拟合的观测。
* **held-out 2D**：把交会出的三维点投影到**没有参与交会**的帧，与该帧的标签
  比较。这是标签精度，比较的是同一渲染像素系（跨域不可比）。
* **random baseline**：同样的交会与评测代码，但每次从候选帧里随机抽同样多的
  视角，重复 `TRIALS` 次。报告的是“试验中位数的中位/均值”，以及我们赢的
  比例和配对差——随机基线本身也在波动，所以不能只看单次。
* **3D vs GT（Sim(3) LOO）**：SfM 世界没有绝对尺度，只能比形状。对每个点用
  其余点拟合相似变换再预测它（留一），误差单位 mm；`scale` 是拟合出的
  mm/单位。ChArUco 用板面坐标（`description` 里的 `board=(x,y,0)mm`），
  YCBV 用 `points_summary.json` 的 `xyz_mm`。

中位数和均值同时给出：只报中位会把长尾藏起来。

## 第 01–03 步：数据集已有的产物与本机复现

`run_demo.sh` 默认不重跑建图与训练。下面两条命令用于核验这些产物：

```bash
# 01-03 的现状审计（只读）
python3 examples/check_stages.py --scene "$SCENE"

# 02 的可复现性：重新抽子集并去畸变（需要 pycolmap）
python3 -m renderlabel subset "$SCENE" -n 250 --out "$SCENE/../work/subset"
```

在本机（2026-11-12）对这两套数据实测：

* `subset` 抽出的帧与数据集自带的训练子集**逐名一致**（YCBV 300/300，
  ChArUco 250/250），文本模型往返的位姿与 point2D 差为 0；
* 用数据集训练好的 `point_cloud.ply` 现场渲染 2 帧成功（ArUco 1920×1080，
  YCBV 640×480），说明第 03 步的共享针孔渲染可以复跑。

注意共享相机的默认焦距是 **COLMAP 相机焦距的中位数**。数据集自带的
`3dgs_renders` 用的是另外固定的焦距（ArUco 1715，YCBV 1066.778），所以现场
重渲与既有渲染图会有一个很小的径向缩放差；要逐像素复现旧渲染，加
`--fx 1715` / `--fx 1066.778`（并给 `--fy` 同值）。demo 的 04–07 步用的是
数据集既有渲染图及其 `message`，两者自洽，不受这个默认值影响。

## 已知边界（demo 特意保留，不修）

* **ArUco 在 N=4 时中位赢、均值输。** 选中的 4 个视角在留出集上的中位误差
  优于随机（胜率约 87%），但均值略差于随机（胜率约 26%，配对差 +0.075 px，
  见 `demo_results.md`）。4 个视角张得最开，基线长；少数几次点击的噪声会
  变成三维点的系统性偏移，均值能看见它，中位数看不见。这说明“选帧”的收益
  是分布中段的，不是尾部——只报中位会把它说成全面胜利。
* **YCBV 的种子对只共享 2 个控制点。** 位姿代理选出的两帧相隔 133°，正好看到
  物体的两侧；19 个关键点里只有 2 个在两张图上都可见。于是种子阶段的 X0
  只由 2 个点定出，`N_sat` 给 7；用第 07 步 18 个点交会出的模型重算，
  `Theta_span` 降到 131.16°，`N_sat=6`。这是冷启动只允许看位姿的代价，
  不是实现错误——最终标签仍然正常。真实采集时如果两帧共视点太少，换一对
  种子帧即可（`--frames` 显式指定）。
* **两次点击的点不可验证。** YCBV 的 `kp14` 只在选中帧里被看到 2 次，两点
  交会的残差已达 7.6 px，留出集误差中位 9.3 px。`--min-views 3` 会剔除它
  （以及另外 4 个本来就不错的 2 点），池化指标并没有变好（见
  `demo_results.md` 最后一节）。demo 默认保留 `min_views=2`，不按结果调参。
* **标签不是绝对真值。** 评测参考是渲染域标签：ArUco 侧来自检测器（单帧
  自洽约 0.7 px），YCBV 侧来自光流搬运（中位误差约 1.3 px）。所以留出集误差
  有一个与视角数无关的地板；demo 里随机基线也吃同一个地板，比较的是选帧
  策略的差，不是“零误差”。
* **跨域不可比。** 这里的像素数只在各自渲染域的共享针孔下有意义，不能与
  真实图像上的误差直接比较。

## 结果

`examples/demo_results.md` 由 `summarize.py` 从两次运行的 JSON 生成；每次重跑
demo 后重新生成即可，不要手抄数字。
