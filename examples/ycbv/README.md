# YCBV demo

用 YCBV 自建渲染集的 `seq00_room_20261012`（1000 帧、19 个物体关键点），跑
第 04–07 步。物体到相机约 500 mm，渲染图 640×480，共享针孔
`fx=fy=1066.778, cx=312.99, cy=241.31`。

## 数据布局

```
<SCENE>/
├── images/                        1000 张原始渲染图
├── colmap/sparse/0/               COLMAP 模型（SIMPLE_RADIAL，1000 个相机）
├── pose/                          GT 位姿（本 demo 不使用）
├── points_summary.json            19 个关键点的真值三维坐标（mm）
├── 3dgs/split_300/                训练子集（300 帧）+ output/point_cloud/iteration_30000
├── 3dgs_renders/
│   ├── images/                    1000 张渲染图
│   └── message/                   1000 个位姿 JSON + _intrinsics.json + _poses.json
└── labels_render/
    ├── points_states.csv          逐 (frame,kp) 的渲染域标签
    └── summary.json               标签生成过程的统计
```

`points_states.csv` 的关键列：

* `u_px, v_px`：**渲染图**上的二维标签（光流从真实图像搬过来，中位误差约
  1.3 px）；只在 `state=visible` 时可信；
* `u_real_px, v_real_px`：真实图像上的 GT 投影（本 demo 不用，因为像素系不同）；
* `state`：`visible` / `invisible`；`invisible` 的行虽然也写了坐标，但那是
  外推值，`make_clicks.py` 会跳过。

每帧可见 8–14 个点（中位 11），所以第 07 步里有些点在某些帧没有点击是正常的，
`triangulate` 会按“至少 2 次点击”处理并报告哪些点失败。

## 运行

```bash
SCENE=/path/to/ycbv_little/data/train/seq00_room_20261012 \
    bash examples/ycbv/run_demo.sh
```

只读核验第 01–03 步：

```bash
python3 examples/check_stages.py --scene "$SCENE" \
    --sparse "$SCENE/colmap/sparse/0" --subdir 3dgs/split_300 \
    --renders "$SCENE/3dgs_renders"
```

## 看点

* 位姿代理选的种子对相隔 133°，正好看到物体的两侧；19 个点里只有 2 个在两帧
  都可见。种子 X0 因此较粗，`N_sat` 给 7；用第 07 步的 18 点模型重算，`N_sat`
  是 6。这就是“冷启动只看位姿”的代价，demo 故意保留（见
  [`../README.md`](../README.md#已知边界demo-特意保留不修)）。
* YCBV 有真实的景深和自遮挡，候选帧始终是全部 1000 帧（目标始终在画面内），
  所以这里比较的是“视角覆盖”，不是“先筛可见帧”。
* 第 07 步会有一个点（本例 `kp15`）因为选中帧里可见次数不足 2 而失败；
  `points3d.csv` 里只有 18 个点，`summary.json` 的 `failed_points` 会记录它。
  这是数据本身的可见性限制，不是 bug。
* 另一个点 `kp14` 只有 2 次点击，残差 7.6 px，留出集误差中位 9.3 px——两点
  交会没有多余观测可以拒绝坏点。用 `MIN_VIEWS=3` 可以把这类点剔除，但池化
  指标不会因此变好（同时会剔掉几个本来不错的 2 点），见
  [`../demo_results.md`](../demo_results.md)。demo 默认不按结果调参。

参考运行的汇总见 [`../demo_results.md`](../demo_results.md)。

图册与演示视频：`work/gallery/report.html`、`work/gallery/demo.mp4`
（由 `visualize.py` / `make_demo_video.py` 生成，命令见
[`../README.md`](../README.md#图册与演示视频)）。
