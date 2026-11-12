# ArUco（ChArUco）demo

用一块 ChArUco 标定板的 946 帧环绕拍摄，跑第 04–07 步：位姿代理选种子对、
两视图交会、预测饱和视图数并选帧、交会 15 个板面控制点、把标签投影回全部
渲染图。

## 数据布局

```
<SCENE>/
├── images/                 946 张原始图（1920×1080）
├── sparse/0/               COLMAP 模型（SIMPLE_RADIAL，逐图一个相机）
└── 3dgs/
    ├── images/             训练用的 250 帧（subset 的输出）
    ├── sparse/0/           去畸变后的 PINHOLE 模型
    ├── output/point_cloud/iteration_30000/point_cloud.ply
    └── renders/
        ├── images/         946 张共享针孔渲染图（fx=fy=1715, cx=960, cy=540）
        ├── message/        每帧一个 JSON（位姿 + 内参）
        └── labels/         每帧一个 labelme JSON：15 个控制点 + 1 个板矩形
```

`labels/` 是在**渲染图上直接检测** ChArUco 标志得到的（沿边剖面 + 抛物线
插值 + 直线拟合 + 邻边求交），不是本项目投影出来的，所以可以用来独立评测。
15 个点：`m0_c0..m0_c3`、`m1_c0..m1_c3`（两个 marker 的角点）、
`origin`、四边中点 `mid_*`、对角 `corner_bl/corner_tr`。每个 shape 的
`description` 给出板面坐标，例如 `board=(-70,70,0)mm`，评测时直接读它当真值。

## 运行

```bash
SCENE=/path/to/charuco/train/0000 bash examples/charuco/run_demo.sh
```

只读核验第 01–03 步：

```bash
python3 examples/check_stages.py --scene "$SCENE"
```

## 看点

* 种子对（位姿代理的最远一对，本数据是 81.6°）在两帧上都有全部 15 个点，
  所以种子交会残差已经在亚像素量级。
* `Theta_span=74.94°`、`theta_eff=11.5°` → `N_sat=4`；贪心在 5° 守卫下正好
  选出 4 帧，覆盖半径 `theta_max=25.54°`。
* 第 07 步只用了这 4 帧的交会结果，却要投影到全部 946 帧；`summary.json`
  里的 coverage 是每个点在多少帧里落在画面内。
* 评测的留出集是 942 帧（946 减去 4 帧标注帧）。随机基线用同样的 4 个视角
  预算重复 100 次，可以看出“覆盖式选帧”带来的差。
* 结果是**中位赢、均值不赢**：留出集误差中位数优于随机，均值反而略差
  （见 `../demo_results.md`）。4 个张得最开的视角基线最长，少数点击的噪声
  会变成三维点的系统性偏移；中位数看不到这种尾部，均值能看到。这个 demo
  把两种统计量都列出来，就是为了不把“中位更好”说成“全面更好”。

参考运行的汇总见 [`../demo_results.md`](../demo_results.md)（由
`summarize.py` 从 `work/eval.json` 生成）。
