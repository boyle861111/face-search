# -*- coding: utf-8 -*-
r"""
照片人臉辨識（圖形介面版）

執行：雙擊「啟動.bat」，或在 Anaconda Prompt：
    conda activate face
    cd /d D:\face
    python face_app.py

目標資料夾結構：目標\人名\照片.jpg（每人一個子資料夾，3~10 張）
"""
import csv
import json
import os
import queue
import shutil
import sqlite3
import sys
import threading
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, ImageTk

APP_VERSION = "dev"   # 打包時由 build.yml 改寫為版本號或 commit 代碼
BASE = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent


def _pick_data_dir():
    """exe 旁可寫就用 exe 旁；否則（例如放在 Program Files）改用 %LOCALAPPDATA%\\FaceSearch"""
    try:
        t = BASE / ".write_test"
        t.write_text("x", encoding="utf-8")
        t.unlink()
        return BASE
    except OSError:
        root = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        d = Path(root) / "FaceSearch"
        d.mkdir(parents=True, exist_ok=True)
        return d


DATA_DIR = _pick_data_dir()
# 模型隨 exe 附帶；若 exe 旁沒有模型又不可寫，下載到資料夾 DATA_DIR
MODEL_DIR = BASE / "models" / "buffalo_l"
if DATA_DIR != BASE and not (MODEL_DIR / "det_10g.onnx").exists():
    MODEL_DIR = DATA_DIR / "models" / "buffalo_l"
MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
DB_PATH = DATA_DIR / "faces.db"
SETTINGS_PATH = DATA_DIR / "settings.json"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_SIDE = 2560      # 讀圖長邊上限，節省記憶體
MIN_FACE = 20        # 忽略小於此像素的臉
MARGIN = 0.10        # 門檻以下多少範圍列為「邊緣」

ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


class UserError(Exception):
    pass


LOG_PATH = DATA_DIR / "face_search.log"


def friendly_error(e):
    """把非預期例外轉成一般人看得懂的中文說明"""
    import errno
    if isinstance(e, MemoryError) or "out of memory" in str(e).lower() or "bad_alloc" in str(e):
        msg = "記憶體不足。請關閉其他程式，或把「偵測解析度」調低後再試。"
    elif isinstance(e, PermissionError) or getattr(e, "errno", None) in (errno.EACCES, errno.EPERM):
        msg = (f"沒有權限讀寫檔案：{getattr(e, 'filename', '') or ''}\n"
               "請確認資料夾不是唯讀、檔案沒有被其他程式開啟，或改選其他輸出資料夾。")
    elif getattr(e, "errno", None) == errno.ENOSPC or getattr(e, "winerror", None) == 112:
        msg = "磁碟空間不足。請清出空間或改選其他磁碟的輸出資料夾。"
    elif getattr(e, "winerror", None) in (206, 3) or getattr(e, "errno", None) == errno.ENAMETOOLONG:
        msg = (f"路徑太長或找不到路徑：{getattr(e, 'filename', '') or ''}\n"
               "請把照片／輸出資料夾移到較短的路徑（例如 D:\\照片）後再試。")
    elif isinstance(e, FileNotFoundError):
        msg = f"找不到檔案或資料夾：{getattr(e, 'filename', '') or ''}\n請確認資料夾仍存在（外接硬碟是否已拔除）。"
    else:
        msg = f"發生未預期的錯誤：{e}"
    return f"{msg}\n\n（詳細錯誤已記錄在：{LOG_PATH}）"


def log_exception(title):
    """把目前例外的完整 traceback 附加到 log 檔（失敗時忽略）"""
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {title}\n{traceback.format_exc()}")
    except Exception:
        pass


# ================================================================ 模型
def ensure_models(status):
    det = MODEL_DIR / "det_10g.onnx"
    rec = MODEL_DIR / "w600k_r50.onnx"
    if det.exists() and rec.exists():
        return det, rec
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = MODEL_DIR.parent / "buffalo_l.zip"
    if not zip_path.exists():
        tmp = zip_path.with_suffix(".part")

        def hook(blocks, bs, total):
            if total > 0:
                status(f"第一次使用，下載模型中（約 280MB）... {min(blocks * bs, total) / total * 100:.0f}%")
        try:
            urllib.request.urlretrieve(MODEL_URL, tmp, hook)
        except Exception as e:
            raise UserError(f"模型下載失敗：{e}\n請手動下載：\n{MODEL_URL}\n放到：{zip_path}")
        tmp.rename(zip_path)
    status("解壓縮模型中...")
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            fn = os.path.basename(name)
            if fn in ("det_10g.onnx", "w600k_r50.onnx"):
                with z.open(name) as src, open(MODEL_DIR / fn, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    if not (det.exists() and rec.exists()):
        raise UserError(f"模型解壓失敗，請刪除 {zip_path} 後重試。")
    return det, rec


def make_session(path, use_gpu):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    if use_gpu and "DmlExecutionProvider" in ort.get_available_providers():
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        try:
            return ort.InferenceSession(str(path), sess_options=so, providers=providers)
        except Exception:
            # 顯示卡驅動不相容等原因，改用 CPU（呼叫端由 get_providers() 得知實際裝置）
            so = ort.SessionOptions()
            so.log_severity_level = 3
            providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), sess_options=so, providers=providers)


def nms(boxes, scores, thr):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thr)[0] + 1]
    return keep


class Detector:
    """SCRFD (det_10g)：回傳 [(bbox[x1,y1,x2,y2], kps[5,2], score), ...]"""

    def __init__(self, path, use_gpu, det_size=640, thresh=0.5):
        self.sess = make_session(path, use_gpu)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.batched = len(self.sess.get_outputs()[0].shape) == 3
        self.size = det_size
        self.thresh = thresh
        self.strides = [8, 16, 32]
        self.fmc = 3
        self.num_anchors = 2
        self._cache = {}

    def _anchors(self, fh, fw, stride):
        key = (fh, fw, stride)
        if key not in self._cache:
            ac = np.stack(np.mgrid[:fh, :fw][::-1], axis=-1).astype(np.float32)
            ac = (ac * stride).reshape(-1, 2)
            ac = np.stack([ac] * self.num_anchors, axis=1).reshape(-1, 2)
            self._cache[key] = ac
        return self._cache[key]

    def detect(self, img):
        h0, w0 = img.shape[:2]
        S = self.size
        scale = S / max(h0, w0)
        nh, nw = max(1, int(round(h0 * scale))), max(1, int(round(w0 * scale)))
        canvas = np.zeros((S, S, 3), dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
        blob = ((canvas.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
        outs = self.sess.run(self.output_names, {self.input_name: blob})

        all_s, all_b, all_k = [], [], []
        for i, s in enumerate(self.strides):
            sc, bb, kp = outs[i], outs[i + self.fmc], outs[i + self.fmc * 2]
            if self.batched:
                sc, bb, kp = sc[0], bb[0], kp[0]
            sc = sc.reshape(-1)
            bb = bb.reshape(-1, 4) * s
            kp = kp.reshape(-1, 10) * s
            idx = np.where(sc >= self.thresh)[0]
            if idx.size == 0:
                continue
            a = self._anchors(S // s, S // s, s)[idx]
            b, k = bb[idx], kp[idx]
            boxes = np.stack([a[:, 0] - b[:, 0], a[:, 1] - b[:, 1],
                              a[:, 0] + b[:, 2], a[:, 1] + b[:, 3]], axis=1)
            kps = np.empty((len(idx), 5, 2), np.float32)
            kps[:, :, 0] = a[:, 0:1] + k[:, 0::2]
            kps[:, :, 1] = a[:, 1:2] + k[:, 1::2]
            all_s.append(sc[idx]); all_b.append(boxes); all_k.append(kps)
        if not all_s:
            return []
        scores = np.concatenate(all_s)
        boxes = np.concatenate(all_b) / scale
        kps = np.concatenate(all_k) / scale
        return [(boxes[i], kps[i], float(scores[i])) for i in nms(boxes, scores, 0.4)]


class Recognizer:
    """ArcFace (w600k_r50)：回傳 L2 正規化 512 維向量"""

    def __init__(self, path, use_gpu):
        self.sess = make_session(path, use_gpu)
        self.input_name = self.sess.get_inputs()[0].name

        bdim = self.sess.get_inputs()[0].shape[0]
        self.can_batch = not isinstance(bdim, int) or bdim != 1

    @staticmethod
    def _crop(img, kps):
        M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), ARCFACE_DST, method=cv2.LMEDS)
        if M is None:
            return None
        return cv2.warpAffine(img, M, (112, 112), borderValue=0)

    def embed_many(self, img, kps_list):
        """一次計算多張臉；回傳與 kps_list 等長的 list（失敗者為 None）"""
        crops = [self._crop(img, k) for k in kps_list]
        ok = [i for i, c in enumerate(crops) if c is not None]
        out = [None] * len(kps_list)
        if not ok:
            return out
        blob = ((np.stack([crops[i] for i in ok]).astype(np.float32) - 127.5) / 127.5).transpose(0, 3, 1, 2)
        vs = None
        if self.can_batch and len(ok) > 1:
            try:
                vs = self.sess.run(None, {self.input_name: blob})[0]
            except Exception:
                self.can_batch = False   # 此模型/GPU 不支援批次，之後改逐張計算
        if vs is None:
            vs = np.concatenate([self.sess.run(None, {self.input_name: blob[j:j + 1]})[0]
                                 for j in range(len(ok))])
        vs = vs / np.linalg.norm(vs, axis=1, keepdims=True)
        for j, i in enumerate(ok):
            out[i] = vs[j].astype(np.float32)
        return out


# ================================================================ 工具
def load_rgb(path):
    with Image.open(path) as im:
        try:
            im.draft("RGB", (MAX_SIDE, MAX_SIDE))  # JPEG 直接以縮小比例解碼，大幅加速
        except Exception:
            pass
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size) > MAX_SIDE:
            im.thumbnail((MAX_SIDE, MAX_SIDE))
        return np.array(im)


def open_db():
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS photos(
            id INTEGER PRIMARY KEY, path TEXT UNIQUE, mtime REAL, size INTEGER,
            n_faces INTEGER, error TEXT);
        CREATE TABLE IF NOT EXISTS faces(
            id INTEGER PRIMARY KEY, photo_id INTEGER, x1 REAL, y1 REAL, x2 REAL, y2 REAL,
            det_score REAL, emb BLOB);
        CREATE INDEX IF NOT EXISTS ix_faces_photo ON faces(photo_id);
    """)
    if "det_size" not in [r[1] for r in con.execute("PRAGMA table_info(photos)")]:
        con.execute("ALTER TABLE photos ADD COLUMN det_size INTEGER")
        con.commit()
    return con


def is_under(p, d):
    try:
        p.relative_to(d)
        return True
    except ValueError:
        return False


def copy_unique(src, folder, score=None):
    folder.mkdir(parents=True, exist_ok=True)
    name = src.name if score is None else f"{score:.3f}_{src.name}"
    dst = folder / name
    stem, suf, k = dst.stem, dst.suffix, 1
    while dst.exists():
        dst = folder / f"{stem}_{k}{suf}"
        k += 1
    shutil.copy2(src, dst)
    return dst


def fmt_time(sec):
    sec = int(sec)
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


# ================================================================ 搜尋工作（背景執行緒）
class Job:
    def __init__(self, cfg, q):
        self.cfg = cfg
        self.q = q
        self.stop = threading.Event()

    def log(self, msg):
        self.q.put(("log", msg))

    def status(self, msg):
        self.q.put(("status", msg))

    def run(self):
        try:
            self._run()
        except UserError as e:
            self.q.put(("error", str(e)))
        except Exception as e:
            log_exception("執行時發生非預期錯誤")
            self.q.put(("error", friendly_error(e)))
        finally:
            self.q.put(("finished", None))

    def _embed_targets(self, det, rec):
        """
        每張目標照片可能有多張臉：
        1) 先用「只有一張臉」的照片建立參考
        2) 多張臉的照片，挑與參考最像的那張臉（沒有參考時取最大臉）
        3) 樣本 ≥ 3 張時，剔除與其他樣本明顯不像的離群樣本
        """
        targets = {}
        troot = Path(self.cfg["targets"])
        dirs = sorted(x for x in troot.iterdir() if x.is_dir())
        if not dirs and any(p.suffix.lower() in EXTS for p in troot.iterdir() if p.is_file()):
            # 目標資料夾直接放照片（沒有人名子資料夾）：整個資料夾當作一位目標
            self.log(f"目標資料夾內沒有人名子資料夾，已把裡面的照片當作同一位目標「{troot.name}」。"
                     "若要找多個人，請為每個人建立一個子資料夾（目標\\人名\\照片.jpg）。")
            dirs = [troot]
        for d in dirs:
            singles, multis = [], []   # (檔名, 向量) / (檔名, [向量...], 最大臉索引)
            for p in sorted(d.iterdir()):
                if p.suffix.lower() not in EXTS:
                    continue
                try:
                    img = load_rgb(p)
                except Exception as ex:
                    self.log(f"[{d.name}] {p.name}：無法讀取（{ex}）")
                    continue
                faces = det.detect(img)
                vecs = rec.embed_many(img, [k for _, k, _ in faces])
                pairs = [(f, v) for f, v in zip(faces, vecs) if v is not None]
                if not pairs:
                    self.log(f"[{d.name}] {p.name}：沒有偵測到人臉，略過")
                elif len(pairs) == 1:
                    singles.append((p.name, pairs[0][1]))
                else:
                    areas = [(f[0][2] - f[0][0]) * (f[0][3] - f[0][1]) for f, _ in pairs]
                    multis.append((p.name, [v for _, v in pairs], int(np.argmax(areas))))

            samples = list(singles)
            for name, vs, largest in multis:
                if singles:
                    ref = np.mean([v for _, v in singles], axis=0)
                    sims = [float(v @ ref) for v in vs]
                    pick = int(np.argmax(sims))
                    self.log(f"[{d.name}] {name}：有 {len(vs)} 張臉，選擇與其他樣本最像的那張（相似度 {sims[pick]:.2f}）")
                else:
                    pick = largest
                    self.log(f"[{d.name}] {name}：有 {len(vs)} 張臉，取最大的那張（建議放幾張只有本人的照片）")
                samples.append((name, vs[pick]))

            if len(samples) >= 3:
                T = np.stack([v for _, v in samples])
                sim = T @ T.T
                np.fill_diagonal(sim, np.nan)
                avg = np.nanmean(sim, axis=1)
                keep = avg >= 0.2
                if keep.sum() >= 2:
                    for (name, _), k, a in zip(samples, keep, avg):
                        if not k:
                            self.log(f"[{d.name}] {name}：與其他樣本平均相似度僅 {a:.2f}，判定不是同一人，已排除")
                    samples = [s for s, k in zip(samples, keep) if k]

            if samples:
                T = np.stack([v for _, v in samples])
                msg = f"目標「{d.name}」：{len(samples)} 張樣本"
                if len(samples) > 1:
                    sim = T @ T.T
                    np.fill_diagonal(sim, np.nan)
                    msg += f"，樣本間最低相似度 {float(np.nanmin(sim)):.2f}"
                if len(samples) < 3:
                    msg += "（樣本少於 3 張，準確度可能較低，建議多放幾張不同角度的清楚照片）"
                self.log(msg)
                targets[d.name] = T
            else:
                self.log(f"目標「{d.name}」無可用樣本（資料夾內沒有照片，或照片都偵測不到人臉），已略過。")
        return targets

    def _compute(self, con, det, rec, p, st):
        embs, err = [], None
        try:
            img = load_rgb(p)
        except Exception as ex:
            img, err = None, str(ex)[:200]
        if img is not None:
            # 推論錯誤（例如 GPU 中途失敗）直接拋出，由呼叫端處理，不記成「讀取失敗」也不寫入快取
            faces = [f for f in det.detect(img) if min(f[0][2] - f[0][0], f[0][3] - f[0][1]) >= MIN_FACE]
            vecs = rec.embed_many(img, [k for _, k, _ in faces])
            embs = [(b, s, e) for (b, _, s), e in zip(faces, vecs) if e is not None]
        with con:
            old = con.execute("SELECT id FROM photos WHERE path=?", (str(p),)).fetchone()
            if old:
                con.execute("DELETE FROM faces WHERE photo_id=?", (old[0],))
                con.execute("DELETE FROM photos WHERE id=?", (old[0],))
            pid = con.execute(
                "INSERT INTO photos(path, mtime, size, n_faces, error, det_size) VALUES (?,?,?,?,?,?)",
                (str(p), st.st_mtime, st.st_size, len(embs), err, self.cfg["det_size"])).lastrowid
            con.executemany(
                "INSERT INTO faces(photo_id, x1, y1, x2, y2, det_score, emb) VALUES (?,?,?,?,?,?,?)",
                [(pid, *map(float, b[:4]), s, e.tobytes()) for b, s, e in embs])
        if err:
            self.log(f"無法讀取：{p}（{err}）")
        return [e for _, _, e in embs]

    def _clean_cache(self, con, pdir, files):
        """移除快取中位於本次照片資料夾、但檔案已不存在的紀錄（其他資料夾的快取不動）"""
        try:
            seen = {str(p) for p in files}
            gone = [(pid,) for pid, path in con.execute("SELECT id, path FROM photos")
                    if path not in seen and is_under(Path(path), pdir) and not Path(path).exists()]
            if gone:
                with con:
                    con.executemany("DELETE FROM faces WHERE photo_id=?", gone)
                    con.executemany("DELETE FROM photos WHERE id=?", gone)
                self.log(f"已清除 {len(gone)} 筆已不存在照片的快取紀錄。")
        except Exception:
            log_exception("快取清理失敗")

    def _run(self):
        cfg = self.cfg
        tdir = Path(cfg["targets"]).resolve()
        pdir = Path(cfg["photos"]).resolve()
        odir = Path(cfg["out"]).resolve()

        self.status("載入模型中...")
        det_p, rec_p = ensure_models(self.status)
        det = Detector(det_p, cfg["gpu"], cfg["det_size"])
        rec = Recognizer(rec_p, cfg["gpu"])
        on_gpu = "DmlExecutionProvider" in det.sess.get_providers()
        dev = "GPU (DirectML)" if on_gpu else "CPU"
        self.log(f"模型載入完成，使用：{dev}")
        if cfg["gpu"] and not on_gpu:
            self.log("提示：無法使用 GPU（可能是顯示卡驅動不相容），已自動改用 CPU，速度會較慢。")

        self.status("處理目標照片...")
        targets = self._embed_targets(det, rec)
        if not targets:
            raise UserError("目標資料夾內沒有可用的照片。\n結構應為：目標\\人名\\照片.jpg")
        names = list(targets)

        self.status("掃描照片清單...")
        files = [p for p in pdir.rglob("*")
                 if p.suffix.lower() in EXTS and p.is_file()
                 and not is_under(p, tdir) and not is_under(p, odir)]
        n = len(files)
        if n == 0:
            raise UserError("照片資料夾內沒有找到照片（支援 jpg / png / bmp / webp）。")
        self.log(f"共 {n} 張照片，開始搜尋...")

        run_dir = odir / time.strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        self.q.put(("run_dir", str(run_dir)))

        con = open_db()
        # 讀取失敗（error 不為空）的紀錄不列入快取，下次會重新分析
        cache = {r[0]: (r[1], r[2], r[3], r[4] or 640)
                 for r in con.execute("SELECT path, id, mtime, size, det_size FROM photos WHERE error IS NULL")}
        thr = cfg["threshold"]
        low = thr - MARGIN
        counts = {nm: [0, 0] for nm in names}
        n_faces = n_new = processed = 0
        t0 = time.time()

        with open(run_dir / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["目標", "分類", "相似度", "原始照片", "輸出檔案"])
            for i, p in enumerate(files, 1):
                if self.stop.is_set():
                    self.log("已停止（已處理的照片已存入快取，下次會更快）。")
                    break
                try:
                    st = p.stat()
                except OSError:
                    continue
                c = cache.get(str(p))
                if c and c[1] == st.st_mtime and c[2] == st.st_size and c[3] == cfg["det_size"]:
                    embs = [np.frombuffer(r[0], np.float32)
                            for r in con.execute("SELECT emb FROM faces WHERE photo_id=?", (c[0],))]
                else:
                    try:
                        embs = self._compute(con, det, rec, p, st)
                    except Exception:
                        if not on_gpu:
                            raise
                        # GPU 推論中途失敗：改用 CPU 重建模型後重試這張
                        log_exception("GPU 推論失敗，改用 CPU")
                        self.log("提示：GPU 運算發生錯誤，已自動改用 CPU 繼續（速度會較慢）。")
                        det = Detector(det_p, False, cfg["det_size"])
                        rec = Recognizer(rec_p, False)
                        on_gpu = False
                        embs = self._compute(con, det, rec, p, st)
                    n_new += 1
                n_faces += len(embs)

                if embs:
                    E = np.stack(embs)
                    for nm in names:
                        best = float((E @ targets[nm].T).max())
                        if best >= thr:
                            cat = "命中"
                        elif cfg["near"] and best >= low:
                            cat = "邊緣"
                        else:
                            continue
                        dst = copy_unique(p, run_dir / nm / cat, best if cfg["score_name"] else None)
                        counts[nm][0 if cat == "命中" else 1] += 1
                        w.writerow([nm, cat, f"{best:.4f}", str(p), str(dst)])
                        f.flush()
                        self.q.put(("hit", (nm, cat, best, str(p), str(dst))))

                processed = i
                el = time.time() - t0
                self.q.put(("progress", (i, n, n_faces, el, el / i * (n - i))))

        stopped = self.stop.is_set()
        if not stopped:
            self._clean_cache(con, pdir, files)
        summary = "；".join(f"{nm}：命中 {a} 張、邊緣 {b} 張" for nm, (a, b) in counts.items())
        head = f"已中途停止（已處理 {processed} / {n} 張）。" if stopped else "完成。"
        self.log(f"{head}{summary}（本次新分析 {n_new} 張，其餘使用快取，耗時 {fmt_time(time.time() - t0)}）")
        if not stopped and sum(a + b for a, b in counts.values()) == 0:
            self.log("沒有找到任何符合的照片。可以試著：把「相似度門檻」調低一點（例如 0.40），"
                     "或把「偵測解析度」調高（例如 1024），照片中人臉較小時特別有幫助。")
        self.q.put(("done", {"stopped": stopped, "counts": counts}))


# ================================================================ 介面
class Tooltip:
    """滑鼠移到元件上時顯示說明"""

    def __init__(self, widget, text):
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, _=None):
        if self.tip:
            return
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left", background="#ffffe0",
                 relief="solid", borderwidth=1, padx=6, pady=4).pack()

    def _hide(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


def load_settings():
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(d):
    try:
        SETTINGS_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"照片人臉辨識 {APP_VERSION}")
        self.geometry("1150x740")
        self.minsize(950, 600)
        for name in ("TkDefaultFont", "TkTextFont", "TkHeadingFont", "TkMenuFont"):
            try:
                from tkinter import font
                font.nametofont(name).configure(family="Microsoft JhengHei UI", size=10)
            except Exception:
                pass

        s = load_settings()
        self.v_targets = tk.StringVar(value=s.get("targets", str(BASE / "目標")))
        self.v_photos = tk.StringVar(value=s.get("photos", str(BASE / "測試照片")))
        self.v_out = tk.StringVar(value=s.get("out", str(BASE / "結果")))
        self.v_thr = tk.DoubleVar(value=s.get("threshold", 0.50))
        self.v_det = tk.StringVar(value=str(s.get("det_size", 640)))
        self.v_near = tk.BooleanVar(value=s.get("near", True))
        self.v_scorename = tk.BooleanVar(value=s.get("score_name", False))
        self.v_gpu = tk.BooleanVar(value=s.get("gpu", True))

        self.q = queue.Queue()
        self.job = None
        self.run_dir = None
        self.items = {}
        self._photo = None

        self._build()
        self._thr_changed()
        self.after(100, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------- layout
    def _build(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)

        top = ttk.LabelFrame(frm, text="設定", padding=8)
        top.pack(fill="x")
        rows = [("目標資料夾", self.v_targets), ("照片資料夾", self.v_photos), ("輸出資料夾", self.v_out)]
        for r, (label, var) in enumerate(rows):
            ttk.Label(top, text=label).grid(row=r, column=0, sticky="w", pady=2)
            ttk.Entry(top, textvariable=var).grid(row=r, column=1, sticky="ew", padx=6)
            ttk.Button(top, text="瀏覽...", command=lambda v=var: self._browse(v)).grid(row=r, column=2)
        top.columnconfigure(1, weight=1)

        opt = ttk.Frame(top)
        opt.grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))
        tip_thr = ("相似度門檻：越高越嚴格、誤判越少，但可能漏掉角度不同的照片。\n"
                   "建議 0.45～0.55（預設 0.50）。找不到人時調低，誤判太多時調高。")
        tip_det = ("偵測解析度：越高越能找到照片中較小的臉（例如團體照），但速度較慢。\n"
                   "一般照片用 640；團體照、遠景照建議 1024 或 1280。")
        w = ttk.Label(opt, text="相似度門檻 ⓘ")
        w.pack(side="left")
        Tooltip(w, tip_thr)
        w = ttk.Scale(opt, from_=0.25, to=0.70, variable=self.v_thr, length=200, command=self._thr_changed)
        w.pack(side="left", padx=4)
        Tooltip(w, tip_thr)
        self.lbl_thr = ttk.Label(opt, width=12)
        self.lbl_thr.pack(side="left")
        ttk.Label(opt, text="越高越嚴格", foreground="gray").pack(side="left")
        w = ttk.Label(opt, text="偵測解析度 ⓘ")
        w.pack(side="left", padx=(12, 4))
        Tooltip(w, tip_det)
        w = ttk.Combobox(opt, textvariable=self.v_det, values=["640", "800", "1024", "1280"],
                         width=6, state="readonly")
        w.pack(side="left")
        Tooltip(w, tip_det)
        ttk.Label(opt, text="越高越能找到小臉，但較慢", foreground="gray").pack(side="left", padx=(4, 0))
        opt2 = ttk.Frame(top)   # 選項放第二列，避免視窗太窄時被截掉
        opt2.grid(row=4, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Checkbutton(opt2, text="輸出邊緣照片", variable=self.v_near).pack(side="left")
        ttk.Checkbutton(opt2, text="檔名加上相似度", variable=self.v_scorename).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(opt2, text="使用 GPU", variable=self.v_gpu).pack(side="left", padx=(12, 0))

        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=8)
        self.b_start = ttk.Button(btns, text="▶ 開始搜尋", command=self._start)
        self.b_start.pack(side="left")
        self.b_stop = ttk.Button(btns, text="■ 停止", command=self._stop, state="disabled")
        self.b_stop.pack(side="left", padx=6)
        self.b_open = ttk.Button(btns, text="開啟結果資料夾", command=self._open_out, state="disabled")
        self.b_open.pack(side="left")
        self.lbl_count = ttk.Label(btns, text="")
        self.lbl_count.pack(side="right")

        self.pb = ttk.Progressbar(frm, mode="determinate")
        self.pb.pack(fill="x")
        self.lbl_status = ttk.Label(frm, text="就緒")
        self.lbl_status.pack(anchor="w", pady=(2, 0))

        body = ttk.PanedWindow(frm, orient="horizontal")
        body.pack(fill="both", expand=True, pady=(8, 0))

        left = ttk.Frame(body)
        body.add(left, weight=3)
        tv_frame = ttk.Frame(left)
        tv_frame.pack(fill="both", expand=True)
        cols = ("target", "cat", "score", "file")
        self.tree = ttk.Treeview(tv_frame, columns=cols, show="headings", selectmode="browse")
        for c, t, wd in zip(cols, ("目標", "分類", "相似度", "照片"), (100, 60, 70, 360)):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=wd, anchor="w" if c == "file" else "center", stretch=(c == "file"))
        self.tree.tag_configure("命中", foreground="#1a7f37")
        self.tree.tag_configure("邊緣", foreground="#b26a00")
        sb = ttk.Scrollbar(tv_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._preview)
        self.tree.bind("<Double-1>", self._open_selected)

        ttk.Label(left, text="訊息").pack(anchor="w", pady=(6, 0))
        self.log_box = tk.Text(left, height=7, state="disabled", wrap="word")
        self.log_box.pack(fill="x")

        right = ttk.Frame(body, padding=(8, 0, 0, 0))
        body.add(right, weight=2)
        self.preview = ttk.Label(right, text="點選左側結果可預覽\n雙擊可開啟照片", anchor="center", justify="center")
        self.preview.pack(fill="both", expand=True)
        self.lbl_path = ttk.Label(right, text="", wraplength=380, foreground="#666")
        self.lbl_path.pack(fill="x", pady=(4, 0))

    # ---------------- actions
    def _thr_changed(self, *_):
        t = self.v_thr.get()
        self.lbl_thr["text"] = f"{t:.2f}（邊緣 {max(0, t - MARGIN):.2f}）"

    def _browse(self, var):
        d = filedialog.askdirectory(initialdir=var.get() if Path(var.get()).exists() else str(BASE))
        if d:
            var.set(str(Path(d)))

    def _log(self, msg):
        self.log_box["state"] = "normal"
        self.log_box.insert("end", time.strftime("%H:%M:%S ") + msg + "\n")
        self.log_box.see("end")
        self.log_box["state"] = "disabled"

    def _cfg(self):
        return {
            "targets": self.v_targets.get().strip(),
            "photos": self.v_photos.get().strip(),
            "out": self.v_out.get().strip(),
            "threshold": round(self.v_thr.get(), 2),
            "det_size": int(self.v_det.get()),
            "near": self.v_near.get(),
            "score_name": self.v_scorename.get(),
            "gpu": self.v_gpu.get(),
        }

    def _start(self):
        cfg = self._cfg()
        if not Path(cfg["targets"]).is_dir():
            return messagebox.showwarning("提醒", "目標資料夾不存在。")
        if not Path(cfg["photos"]).is_dir():
            return messagebox.showwarning("提醒", "照片資料夾不存在。")
        if not cfg["out"]:
            return messagebox.showwarning("提醒", "請指定輸出資料夾。")
        save_settings(cfg)
        self.tree.delete(*self.tree.get_children())
        self.items.clear()
        self.n_hit = self.n_near = 0
        self.lbl_count["text"] = ""
        self.pb["value"] = 0
        self.run_dir = None
        self.b_open["state"] = "disabled"
        self.b_start["state"] = "disabled"
        self.b_stop["state"] = "normal"
        self.job = Job(cfg, self.q)
        threading.Thread(target=self.job.run, daemon=True).start()

    def _stop(self):
        if self.job:
            self.job.stop.set()
            self.lbl_status["text"] = "停止中..."

    def _open_out(self):
        if self.run_dir and Path(self.run_dir).exists():
            os.startfile(self.run_dir)

    def _selected_path(self):
        sel = self.tree.selection()
        return self.items.get(sel[0]) if sel else None

    def _open_selected(self, _=None):
        p = self._selected_path()
        if p and Path(p).exists():
            os.startfile(p)

    def _preview(self, _=None):
        p = self._selected_path()
        if not p:
            return
        try:
            w = max(200, self.preview.winfo_width())
            h = max(200, self.preview.winfo_height())
            side = max(w, h)
            with Image.open(p) as im:
                try:
                    im.draft("RGB", (side, side))   # JPEG 直接縮小解碼，大照片不會卡頓
                except Exception:
                    pass
                im.thumbnail((side, side), Image.Resampling.BILINEAR)  # 先縮到預覽尺寸再轉正，減少運算量
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((w, h))
                self._photo = ImageTk.PhotoImage(im)
            self.preview.configure(image=self._photo, text="")
        except Exception as e:
            self.preview.configure(image="", text=f"無法預覽：{e}")
        self.lbl_path["text"] = p

    def _on_close(self):
        if self.job and self.b_stop["state"] == "normal":
            if not messagebox.askyesno("確認", "搜尋進行中，確定要關閉嗎？\n（已處理的部分會保留）"):
                return
            self.job.stop.set()
        self.destroy()

    def _on_done(self, data):
        stopped, counts = data["stopped"], data["counts"]
        parts = "｜".join(f"{nm} 命中 {a}／邊緣 {b}" for nm, (a, b) in counts.items())
        self.lbl_status["text"] = f"{'已停止' if stopped else '完成'}｜{parts}"
        total = sum(a + b for a, b in counts.values())
        if total and self.run_dir:
            title = "已停止" if stopped else "搜尋完成"
            if messagebox.askyesno(title, f"{parts.replace('｜', chr(10))}\n\n要開啟結果資料夾嗎？"):
                self._open_out()
        elif not stopped:
            messagebox.showinfo("搜尋完成", "沒有找到符合的照片。\n\n建議：把「相似度門檻」調低一點（例如 0.40），"
                                "或把「偵測解析度」調高（例如 1024）後再試一次。")

    # ---------------- 背景訊息
    def _poll(self):
        try:
            for _ in range(500):
                kind, data = self.q.get_nowait()
                if kind == "log":
                    self._log(data)
                elif kind == "status":
                    self.lbl_status["text"] = data
                elif kind == "progress":
                    i, n, nf, el, eta = data
                    self.pb["maximum"] = n
                    self.pb["value"] = i
                    self.lbl_status["text"] = (f"{i} / {n} 張（{i / n * 100:.0f}%）｜已偵測 {nf} 張臉｜"
                                               f"經過 {fmt_time(el)}｜剩餘約 {fmt_time(eta)}")
                elif kind == "hit":
                    nm, cat, score, src, dst = data
                    iid = self.tree.insert("", "end", values=(nm, cat, f"{score:.3f}", src), tags=(cat,))
                    self.items[iid] = dst
                    if cat == "命中":
                        self.n_hit += 1
                    else:
                        self.n_near += 1
                    self.lbl_count["text"] = f"命中 {self.n_hit}　邊緣 {self.n_near}"
                elif kind == "run_dir":
                    self.run_dir = data
                    self.b_open["state"] = "normal"
                elif kind == "done":
                    self._on_done(data)
                elif kind == "error":
                    self._log("錯誤：" + data.splitlines()[0])
                    messagebox.showerror("錯誤", data[:2000])
                elif kind == "finished":
                    self.b_start["state"] = "normal"
                    self.b_stop["state"] = "disabled"
                    if self.lbl_status["text"].startswith("停止中"):
                        self.lbl_status["text"] = "已停止"
        except queue.Empty:
            pass
        except Exception as e:
            self._log(f"介面錯誤：{e}")
        finally:
            self.after(100, self._poll)


def main():
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    App().mainloop()


if __name__ == "__main__":
    main()
