Python 3.13.2 (v3.13.2:4f8bb3947cf, Feb  4 2025, 11:51:10) [Clang 15.0.0 (clang-1500.3.9.4)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>> #!/usr/bin/env python3
... import sys, subprocess
... from pathlib import Path
... import tempfile, logging, argparse
... from tkinter import Tk, filedialog, messagebox
... import shutil
... import numpy as np
... from PIL import Image
... import torch
... import torchvision.transforms as T
... import skimage.metrics as metrics
... 
... logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
... 
... for pkg, imp in [('numpy','numpy'),('Pillow','PIL'),('scikit-image','skimage'),('torch','torch'),('torchvision','torchvision'),('imageio-ffmpeg','imageio_ffmpeg')]:
...     try:
...         __import__(imp)
...     except ImportError:
...         subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple', '--default-timeout=100'])
... 
... COLMAP_BIN = r'D:\colmap\bin\colmap.exe'
... 
... def run_cmd(cmd, cwd=None):
...     if cmd[0] == 'colmap':
...         cmd[0] = COLMAP_BIN
...     logging.info('Running: ' + ' '.join(cmd))
...     subprocess.run(cmd, cwd=cwd, check=True)
... 
... def colmap_process(images, colmap_dir, gaussian_repo):
...     colmap_dir.mkdir(parents=True, exist_ok=True)
...     db = colmap_dir / 'database.db'
...     run_cmd(['colmap', 'feature_extractor', '--database_path', str(db), '--image_path', str(images)])
    run_cmd(['colmap', 'exhaustive_matcher', '--database_path', str(db)])
    sparse = colmap_dir / 'sparse'
    sparse.mkdir(exist_ok=True)
    run_cmd(['colmap', 'mapper', '--database_path', str(db), '--image_path', str(images), '--output_path', str(sparse)])
    transforms = colmap_dir / 'transforms.json'
    script = Path(gaussian_repo) / 'scripts' / 'colmap2nerf.py'
    if not script.is_file():
        candidates = list(Path(gaussian_repo).rglob('colmap2nerf.py'))
        if candidates:
            script = candidates[0]
            logging.info(f'use {script}')
        else:
            messagebox.showerror('错误', f'not find: {script}')
            sys.exit(1)
    run_cmd([sys.executable, str(script), '--colmap_folder', str(sparse), '--out', str(transforms)])
    return transforms

def train_gaussian(repo, transforms, ws):
    ws.mkdir(parents=True, exist_ok=True)
    run_cmd([sys.executable, 'train.py', '--workspace', str(ws), '--data', str(transforms)], cwd=str(repo))

def render_gaussian(repo, ws, out_video, render_path=None):
    cmd = [sys.executable, 'render.py', '--workspace', str(ws)]
    if render_path:
        cmd += ['--render_path', str(render_path)]
    cmd += ['--video', str(out_video)]
    run_cmd(cmd, cwd=str(repo))

def train_nerf(repo, transforms, out_dir, accel=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, 'run_nerf.py', '--config', str(transforms), '--out_dir', str(out_dir)]
    if accel:
        cmd.append('--acceleration')
    run_cmd(cmd, cwd=str(repo))

def evaluate(gt_dir, pred_dir):
    psnrs, ssims = [], []
    for gt in sorted(gt_dir.glob('*')):
        pr = pred_dir / gt.name
        if not pr.exists(): continue
        g = np.array(Image.open(gt)).astype(np.float32)/255
        p = np.array(Image.open(pr)).astype(np.float32)/255
        psnrs.append(metrics.peak_signal_noise_ratio(g, p, data_range=1.0))
        ssims.append(metrics.peak_signal_similarity(g, p, multichannel=True))
    return {'psnr': float(np.mean(psnrs)), 'ssim': float(np.mean(ssims))}

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gaussian_repo', default='gaussian_splatting')
    parser.add_argument('--nerf_repo', default='nerf_pytorch')
    parser.add_argument('--test_images', default=None)
    parser.add_argument('--output', default='output')
    parser.add_argument('--render_path', default=None)
    args = parser.parse_args()
    root = Tk(); root.withdraw()
    messagebox.showinfo('input', 'choose training video or cancel and choose training images')
    vid = filedialog.askopenfilename(title='video', filetypes=[('video','*.mp4 *.avi *.mov')])
    if vid:
        tmp = Path(tempfile.mkdtemp(prefix='frames_'))
        ff = __import__('imageio_ffmpeg').get_ffmpeg_exe()
        run_cmd([ff, '-i', vid, str(tmp/'frame_%04d.png')])
        images = tmp
    else:
        messagebox.showinfo('input', 'choose training images folder')
        d1 = filedialog.askdirectory(title='train images') or sys.exit()
        images = Path(d1)
    if not args.test_images:
        messagebox.showinfo('input', 'choose test video or cancel and choose test images')
        tvid = filedialog.askopenfilename(title='test video', filetypes=[('video','*.mp4 *.avi *.mov')])
        if tvid:
            tmp2 = Path(tempfile.mkdtemp(prefix='test_'))
            ff2 = __import__('imageio_ffmpeg').get_ffmpeg_exe()
            run_cmd([ff2, '-i', tvid, str(tmp2/'frame_%04d.png')])
            test_images = tmp2
        else:
            d2 = filedialog.askdirectory(title='test images') or sys.exit()
            test_images = Path(d2)
    root.destroy()
    base = Path(args.output); base.mkdir(exist_ok=True)
    transforms = colmap_process(images, base/'colmap', args.gaussian_repo)
    gs_ws = base/'gaussian'; train_gaussian(Path(args.gaussian_repo), transforms, gs_ws)
    gs_vid = base/'gaussian.mp4'; render_gaussian(Path(args.gaussian_repo), gs_ws, gs_vid, Path(args.render_path) if args.render_path else None)
    nb = base/'nerf_base'; train_nerf(Path(args.nerf_repo), transforms, nb, False)
    na = base/'nerf_acc'; train_nerf(Path(args.nerf_repo), transforms, na, True)
    ti = Path(args.test_images) if args.test_images else test_images
    res = {'gaussian':evaluate(ti, gs_ws/'renders'),'nerf_base':evaluate(ti, nb/'renders'),'nerf_acc':evaluate(ti, na/'renders')}
    logging.info('=== results ===')
    for k,v in res.items(): logging.info(f"{k}: PSNR={v['psnr']:.2f}, SSIM={v['ssim']:.4f}")
