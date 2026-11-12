# 02 — Subset, undistortion, and 3DGS training

**Script:** `scripts/02_train_3dgs.sh`
**Input:** `<SCENE>/images/`, `<SCENE>/sparse/0/`
**Output:** `<SCENE>/3dgs/{images,sparse/0}` and
`<SCENE>/3dgs/output/point_cloud/iteration_<ITER>/point_cloud.ply`

## Why a separate 3dgs/ folder

Stock 3DGS accepts only `PINHOLE` / `SIMPLE_PINHOLE` camera models. A COLMAP
model built with the default `SIMPLE_RADIAL` (one camera per image) is rejected
outright. This step:

1. keeps `N_FRAMES` equally spaced **registered** frames,
2. copies the COLMAP text model verbatim and keeps only those image rows (ids
   and point2D ordering never shift),
3. runs COLMAP's image undistorter, which produces PINHOLE cameras and
   undistorted images,
4. lays the result out as `<SCENE>/3dgs/{images,sparse/0}`.

The originals under `<SCENE>/images` and `<SCENE>/sparse` are untouched.

## Environment setup (once)

Two conda environments are used: `rlabel` for the label side (see the README)
and `gs` for training/rendering.

```bash
conda create -n gs python=3.10 -y
conda activate gs

# pick the CUDA wheel that matches your driver
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install plyfile tqdm
```

The rasteriser compiles CUDA kernels; `nvcc --version` must match the CUDA
version of the torch wheel. For an RTX 3090/4090 set the architecture
explicitly if the build targets the wrong one:

```bash
export TORCH_CUDA_ARCH_LIST="8.6"     # 3090; 8.9 for 4090, 8.0 for A100
pip install --force-reinstall submodules/diff-gaussian-rasterization
```

Check:

```bash
python -c "import torch, diff_gaussian_rasterization; print(torch.__version__, torch.cuda.is_available())"
```

## Run

```bash
SCENE=/data/box1 bash scripts/02_train_3dgs.sh
```

which is:

```bash
python -m renderlabel subset $SCENE -n 250
cd $GS_REPO
python train.py -s $SCENE/3dgs -m $SCENE/3dgs/output --iterations 30000
```

### Frame selection

`-n 250` picks 250 frames at even indices of the **sorted image names**. This
assumes names sort in capture order (`frame_00001.png`, `frame_00002.png`, ...).
If they do not, pass an explicit list:

```bash
python -m renderlabel subset $SCENE --names-file my_frames.txt
```

A subset rather than all frames keeps training and rendering time bounded; 200–300
frames is usually plenty for an object capture. The subset is also what gets
rendered later (step 03 renders **every** pose in `sparse/0`, but the 3DGS has
only seen the subset).

## Checks

```bash
ls $SCENE/3dgs/images | wc -l          # == N_FRAMES
colmap model_analyzer --path $SCENE/3dgs/sparse/0   # camera model must be PINHOLE
ls $SCENE/3dgs/output/point_cloud/iteration_30000/point_cloud.ply
```

The subset stage already asserts that the pose and point2D values are unchanged
by the text round-trip, and that the undistorted model has exactly `N_FRAMES`
images.

## Troubleshooting

| symptom | fix |
|---|---|
| `pycolmap is required` | install it into the label env: `conda install -c conda-forge pycolmap`. |
| `Camera model SIMPLE_RADIAL not supported` during training | you passed the wrong source to `train.py`; it must be `-s $SCENE/3dgs`, not `$SCENE`. |
| `Could not recognize scene type!` | `$SCENE/3dgs/sparse/0` is missing or empty; rerun the subset stage. |
| rasteriser build error | torch/CUDA mismatch. Reinstall torch with the CUDA version your `nvcc` provides, then rebuild the submodules. |
| `GaussianRasterizationSettings.__new__() got an unexpected keyword argument 'antialiasing'` | the `diff_gaussian_rasterization` actually imported is not the one built from this repo's submodule (a copy from another project shadows it). Check `python -c "import diff_gaussian_rasterization as d; print(d.__file__)"` and reinstall: `pip install --force-reinstall submodules/diff-gaussian-rasterization`. |
| CUDA out of memory at 1080p | reduce `N_FRAMES`, or lower the resolution of the input images; 3DGS memory scales with image size. |
| training is slower than expected | check `nvidia-smi`; the official code uses most of one GPU. 30k iterations on 250 frames is typically 30–60 min on a modern card. |

Next: [03 — render all poses](03_render.md)
