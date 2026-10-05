#!/usr/bin/env python3
"""
Interactive front end for the Cricut mat aligner.

  python gui.py [photo ...]

- Open one or more photos (file picker, or pass them on the command line).
- Align: automatic grid fit; the fitted 1" lattice is drawn over the photo so
  you can check it sits on the printed mat lines.
- If the fit is wrong (or fails), drag the four corner handles onto the outer
  corners of the 12x12" grid and click "Re-fit from corners". Click a handle
  and use the arrow keys (Shift = x10) to nudge it; a magnifier shows the
  selected corner at full resolution.
- Save / Save all write 3600x3600 px, 300 DPI PNGs.
"""

import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import align, save_image, OUTPUT_SIZE
from grid_fit import (canvas_to_photo, grid_corners_homography,
                      CELL, GRID_N, MARGIN)
from orientation import detect_mat_corners

IMAGE_TYPES = [("Images", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp"),
               ("All files", "*")]

PREVIEW_MAX = 2000        # long side of the cached display copy
PAD = 8                   # view padding (screen px)
HANDLE_R = 7              # corner handle radius (screen px)
LOUPE = 240               # magnifier size (screen px)
LOUPE_SRC = 80            # photo px shown across the magnifier

BG = '#262626'
FG_DIM = '#8a8a8a'
GRID_AUTO = '#00e5ff'     # fitted model
GRID_MANUAL = '#ffd400'   # preview from hand-placed corners
GRID_OUTPUT = '#ff3bd4'   # ideal k*300 lines on the output
HANDLE = '#ff453a'
HANDLE_SEL = '#ffffff'

GRID_LO, GRID_HI = MARGIN, MARGIN + GRID_N * CELL
GRID_CORNERS = np.float32([[GRID_LO, GRID_LO], [GRID_HI, GRID_LO],
                           [GRID_HI, GRID_HI], [GRID_LO, GRID_HI]])


def lattice_polylines(n_samples: int = 25):
    """The 13 + 13 ideal grid lines in canvas coords, as (N, 2) arrays."""
    t = np.linspace(GRID_LO, GRID_HI, n_samples)
    lines = []
    for k in range(GRID_N + 1):
        c = np.full_like(t, GRID_LO + k * CELL)
        lines.append(np.stack([t, c], axis=-1))   # horizontal
        lines.append(np.stack([c, t], axis=-1))   # vertical
    return lines


def guess_grid_corners(image: np.ndarray) -> np.ndarray:
    """Starting handles: mat corners pulled in to the grid, else an inset box."""
    mat = detect_mat_corners(image)
    if mat is None:
        h, w = image.shape[:2]
        m = 0.15 * min(w, h)
        return np.float32([[m, m], [w - m, m], [w - m, h - m], [m, h - m]])
    centre = mat.mean(axis=0)
    return np.float32(mat + (centre - mat) / 13.0)   # grid is ~12/13 of the mat


def corners_valid(corners: np.ndarray) -> bool:
    return bool(cv2.isContourConvex(np.float32(corners).reshape(-1, 1, 2)))


class Item:
    """One photo in the list and everything known about it."""

    def __init__(self, path: str):
        self.path = path
        self.image: Optional[np.ndarray] = None    # BGR photo, loaded lazily
        self.result: Optional[dict] = None         # main.align() output
        self.corners: Optional[np.ndarray] = None  # 4 grid corners (photo px)
        self.corners_edited = False
        self.error: Optional[str] = None
        self.busy = False
        self.saved_to: Optional[str] = None

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def default_output(self) -> str:
        return os.path.splitext(self.name)[0] + '_aligned.png'

    def load(self) -> np.ndarray:
        if self.image is None:
            self.image = cv2.imread(self.path)
            if self.image is None:
                raise IOError(f"Could not read image: {self.path}")
        return self.image

    def label(self) -> str:
        if self.busy:
            mark = '…'
        elif self.error:
            mark = '✗'
        elif self.result is None:
            mark = ' '
        elif self.result['warnings']:
            mark = '⚠'
        else:
            mark = '✓'
        return f"{mark}  {self.name}" + ('  (saved)' if self.saved_to else '')


class ImageView(tk.Canvas):
    """Canvas showing an image scaled to fit, with image <-> screen mapping."""

    def __init__(self, master, placeholder: str):
        super().__init__(master, background=BG, highlightthickness=0,
                         width=480, height=480)
        self.placeholder = placeholder
        self.full: Optional[Image.Image] = None
        self.preview: Optional[Image.Image] = None
        self.scale = 1.0
        self.ox = self.oy = 0.0
        self.overlay = None           # callable(view) drawing 'overlay' items
        self._tk_img = None
        self._tk_key = None
        self.bind('<Configure>', lambda e: self.redraw())

    def set_image(self, pil: Optional[Image.Image]):
        self.full = pil
        self.preview = None
        if pil is not None:
            self.preview = pil.copy()
            self.preview.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.BILINEAR)
        self._tk_key = None
        self.redraw()

    def redraw(self):
        self.delete('all')
        w, h = self.winfo_width(), self.winfo_height()
        if self.full is None:
            self.create_text(w / 2, h / 2, text=self.placeholder, fill=FG_DIM,
                             justify='center')
            return
        fw, fh = self.full.size
        self.scale = min((w - 2 * PAD) / fw, (h - 2 * PAD) / fh)
        if self.scale <= 0:
            return
        dw, dh = max(1, round(fw * self.scale)), max(1, round(fh * self.scale))
        self.ox, self.oy = (w - dw) / 2, (h - dh) / 2
        if self._tk_key != (dw, dh):
            self._tk_img = ImageTk.PhotoImage(
                self.preview.resize((dw, dh), Image.BILINEAR))
            self._tk_key = (dw, dh)
        self.create_image(self.ox, self.oy, anchor='nw', image=self._tk_img)
        self.redraw_overlay()

    def redraw_overlay(self):
        self.delete('overlay')
        if self.full is not None and self.overlay is not None:
            self.overlay(self)

    def to_screen(self, pts: np.ndarray) -> np.ndarray:
        return np.asarray(pts, np.float64) * self.scale + (self.ox, self.oy)

    def to_image(self, x: float, y: float) -> np.ndarray:
        return np.array([(x - self.ox) / self.scale, (y - self.oy) / self.scale])


class App(tk.Tk):

    def __init__(self, paths=()):
        super().__init__()
        self.title("Cricut Mat Aligner")
        self.geometry("1400x860")
        self.minsize(900, 560)

        self.items: list[Item] = []
        self.current: Optional[Item] = None
        self.sel_handle: Optional[int] = None
        self.drag = False
        self.jobs: queue.Queue = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self._loupe_img = None

        self.notch = tk.StringVar(value='top')
        self.phase2 = tk.BooleanVar(value=True)
        self.show_grid = tk.BooleanVar(value=True)

        self._build()
        self._bind_keys()
        self._add_paths(paths)
        self._refresh()
        self.after(100, self._poll_jobs)

    # ------------------------------------------------------------------ UI

    def _build(self):
        side = ttk.Frame(self, padding=8)
        side.pack(side='left', fill='y')

        ttk.Label(side, text="Photos").pack(anchor='w')
        lf = ttk.Frame(side)
        lf.pack(fill='both', expand=True, pady=(2, 4))
        self.listbox = tk.Listbox(lf, width=34, activestyle='none',
                                  exportselection=False)
        sb = ttk.Scrollbar(lf, orient='vertical', command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        self.listbox.bind('<<ListboxSelect>>', self._on_select)

        row = ttk.Frame(side)
        row.pack(fill='x')
        ttk.Button(row, text="Open…", command=self.open_files).pack(side='left', expand=True, fill='x')
        self.btn_remove = ttk.Button(row, text="Remove", command=self.remove_current)
        self.btn_remove.pack(side='left', expand=True, fill='x')

        opts = ttk.LabelFrame(side, text="Options", padding=6)
        opts.pack(fill='x', pady=8)
        ttk.Label(opts, text="Triangle notch:").grid(row=0, column=0, sticky='w')
        ttk.Radiobutton(opts, text="Top", value='top', variable=self.notch).grid(row=0, column=1)
        ttk.Radiobutton(opts, text="Bottom", value='bottom', variable=self.notch).grid(row=0, column=2)
        ttk.Checkbutton(opts, text="Lens / mat-bend correction (Phase 2)",
                        variable=self.phase2).grid(row=1, column=0, columnspan=3, sticky='w')
        ttk.Checkbutton(opts, text="Show grid overlay", variable=self.show_grid,
                        command=self._redraw_overlays).grid(row=2, column=0, columnspan=3, sticky='w')

        act = ttk.LabelFrame(side, text="Align", padding=6)
        act.pack(fill='x')
        self.btn_align = ttk.Button(act, text="Align", command=self.align_current)
        self.btn_align.pack(fill='x')
        self.btn_refit = ttk.Button(act, text="Re-fit from corners", command=self.refit_current)
        self.btn_refit.pack(fill='x')
        self.btn_align_all = ttk.Button(act, text="Align all", command=self.align_all)
        self.btn_align_all.pack(fill='x', pady=(6, 0))

        out = ttk.LabelFrame(side, text="Save", padding=6)
        out.pack(fill='x', pady=8)
        self.btn_save = ttk.Button(out, text="Save…", command=self.save_current)
        self.btn_save.pack(fill='x')
        self.btn_save_all = ttk.Button(out, text="Save all to folder…", command=self.save_all)
        self.btn_save_all.pack(fill='x')

        self.info = ttk.Label(side, text="", wraplength=260, justify='left')
        self.info.pack(fill='x', pady=(4, 0))

        status = ttk.Frame(self, padding=(8, 2))
        status.pack(side='bottom', fill='x')
        self.status = ttk.Label(status, text="")
        self.status.pack(side='left')
        self.progress = ttk.Progressbar(status, mode='indeterminate', length=160)

        panes = ttk.PanedWindow(self, orient='horizontal')
        panes.pack(side='left', fill='both', expand=True)
        self.photo_view = ImageView(panes, "Open a photo of a Cricut mat\n(Open… or ⌘O / Ctrl+O)")
        self.output_view = ImageView(panes, "Aligned output appears here")
        panes.add(self.photo_view, weight=1)
        panes.add(self.output_view, weight=1)
        self.photo_view.overlay = self._draw_photo_overlay
        self.output_view.overlay = self._draw_output_overlay

        pv = self.photo_view
        pv.bind('<ButtonPress-1>', self._on_press)
        pv.bind('<B1-Motion>', self._on_drag)
        pv.bind('<ButtonRelease-1>', self._on_release)
        for key, d in (('Left', (-1, 0)), ('Right', (1, 0)), ('Up', (0, -1)), ('Down', (0, 1))):
            pv.bind(f'<{key}>', lambda e, d=d: self._nudge(d, 1))
            pv.bind(f'<Shift-{key}>', lambda e, d=d: self._nudge(d, 10))
        pv.bind('<Escape>', lambda e: self._select_handle(None))

    def _bind_keys(self):
        mod = 'Command' if sys.platform == 'darwin' else 'Control'
        self.bind_all(f'<{mod}-o>', lambda e: self.open_files())
        self.bind_all(f'<{mod}-s>', lambda e: self.save_current())
        self.bind_all('<Return>', lambda e: self.align_current())

    def _refresh(self):
        """Sync list labels, buttons and the info panel with the state."""
        sel = self.listbox.curselection()
        self.listbox.delete(0, 'end')
        for it in self.items:
            self.listbox.insert('end', it.label())
        if sel and sel[0] < len(self.items):
            self.listbox.selection_set(sel[0])

        busy = self.worker is not None
        it = self.current
        has = it is not None and not busy
        done = [i for i in self.items if i.result is not None]

        def state(btn, on):
            btn.state(['!disabled'] if on else ['disabled'])

        state(self.btn_remove, has)
        state(self.btn_align, has)
        state(self.btn_refit, has and it.corners is not None)
        state(self.btn_align_all, bool(self.items) and not busy)
        state(self.btn_save, has and it.result is not None)
        state(self.btn_save_all, bool(done) and not busy)

        if busy:
            self.progress.pack(side='right')
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.pack_forget()

        self.info.configure(text=self._info_text(it), foreground='')

    def _info_text(self, it: Optional[Item]) -> str:
        if it is None:
            return ""
        if it.busy:
            return "Aligning…"
        if it.error:
            return f"✗ {it.error}"
        lines = []
        r = it.result
        if r is not None:
            if r['qa'] is not None:
                qa = r['qa']
                lines.append(f"QA: {qa['lines']}/26 grid lines found in output\n"
                             f"deviation rms {qa['rms']:.1f} px · p95 {qa['p95']:.1f} px "
                             f"· max {qa['max']:.1f} px  (1 px = 1/300\")")
                lines.append(f"Source resolution ≥ {r['src_dpi']:.0f} px/inch")
            if r['fit'] is None and r['H'] is not None:
                lines.append("Rendered from your corners only (no refined fit).")
            lines += [f"⚠ {w}" for w in r['warnings']]
        if it.corners_edited:
            lines.append("Corners moved — click “Re-fit from corners” to apply.")
        elif r is not None and r['H'] is None:
            lines.append("Drag the four handles onto the outer corners of the "
                         "12×12\" grid, then click “Re-fit from corners”.")
        if it.saved_to:
            lines.append(f"Saved: {it.saved_to}")
        return "\n\n".join(lines)

    def _set_status(self, text: str):
        self.status.configure(text=text)

    # --------------------------------------------------------------- files

    def open_files(self):
        paths = filedialog.askopenfilenames(title="Open mat photos", filetypes=IMAGE_TYPES)
        self._add_paths(paths)

    def _add_paths(self, paths):
        known = {os.path.abspath(i.path) for i in self.items}
        new = [Item(p) for p in paths if os.path.abspath(p) not in known]
        if not new:
            return
        self.items += new
        self._refresh()
        idx = self.items.index(new[0])
        self.listbox.selection_clear(0, 'end')
        self.listbox.selection_set(idx)
        self.listbox.see(idx)
        self._show(new[0])

    def remove_current(self):
        if self.current is None or self.current.busy:
            return
        idx = self.items.index(self.current)
        self.items.pop(idx)
        self._refresh()
        if self.items:
            idx = min(idx, len(self.items) - 1)
            self.listbox.selection_set(idx)
            self._show(self.items[idx])
        else:
            self._show(None)

    def _on_select(self, _event=None):
        sel = self.listbox.curselection()
        if sel:
            self._show(self.items[sel[0]])

    def _show(self, it: Optional[Item]):
        self.current = it
        self.sel_handle = None
        photo = output = None
        if it is not None:
            try:
                photo = Image.fromarray(cv2.cvtColor(it.load(), cv2.COLOR_BGR2RGB))
                h, w = it.image.shape[:2]
                self._set_status(f"{it.name} — {w}×{h}")
            except IOError as e:
                it.error = str(e)
                self._set_status(str(e))
            if it.result is not None:
                output = Image.fromarray(cv2.cvtColor(it.result['result'], cv2.COLOR_BGR2RGB))
        else:
            self._set_status("")
        self.photo_view.set_image(photo)
        self.output_view.set_image(output)
        self._refresh()

    # --------------------------------------------------------------- align

    def align_current(self):
        it = self.current
        if it is None or self.worker is not None:
            return
        self._start([(it, None)])

    def refit_current(self):
        it = self.current
        if it is None or it.corners is None or self.worker is not None:
            return
        if not corners_valid(it.corners):
            messagebox.showwarning("Corners", "The four corners must form a convex "
                                   "shape (no crossed edges). Move them onto the "
                                   "outer corners of the grid.")
            return
        self._start([(it, grid_corners_homography(it.corners))])

    def align_all(self):
        if self.worker is None:
            self._start([(it, None) for it in self.items])

    def _start(self, jobs):
        opts = {'enable_phase2': self.phase2.get(), 'notch': self.notch.get()}
        for it, _ in jobs:
            it.busy = True
        self.worker = threading.Thread(target=self._work, args=(jobs, opts), daemon=True)
        self.worker.start()
        self._set_status(f"Aligning {len(jobs)} photo(s)…")
        self._refresh()

    def _work(self, jobs, opts):
        """Worker thread: no Tk calls here, results go through self.jobs."""
        for it, H_init in jobs:
            try:
                r = align(it.load(), H_init=H_init, **opts)
                if r['fit'] is not None:
                    r['fit'] = {'stats': r['fit']['stats']}   # drop the big canvas
                self.jobs.put((it, r, None))
            except Exception as e:     # report, keep going with the batch
                self.jobs.put((it, None, f"{type(e).__name__}: {e}"))
        self.jobs.put(None)

    def _poll_jobs(self):
        try:
            while True:
                msg = self.jobs.get_nowait()
                if msg is None:
                    self.worker = None
                    self._set_status("Done.")
                    self._refresh()
                    continue
                it, r, err = msg
                it.busy = False
                it.error = err
                it.saved_to = None
                if r is not None:
                    it.result = r
                    it.corners_edited = False
                    if r['H'] is not None:
                        it.corners = np.float32(canvas_to_photo(GRID_CORNERS, r['H']))
                    else:
                        it.corners = guess_grid_corners(it.image)
                if it is self.current:
                    self._show(it)
                else:
                    self._refresh()
        except queue.Empty:
            pass
        self.after(100, self._poll_jobs)

    # ---------------------------------------------------------------- save

    def save_current(self):
        it = self.current
        if it is None or it.result is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save aligned image", defaultextension='.png',
            initialdir=os.path.dirname(it.path), initialfile=it.default_output,
            filetypes=[("PNG", "*.png")])
        if path:
            self._save(it, path)
            self._refresh()

    def save_all(self):
        done = [i for i in self.items if i.result is not None]
        if not done:
            return
        folder = filedialog.askdirectory(title="Save aligned images to folder")
        if not folder:
            return
        targets = [(i, os.path.join(folder, i.default_output)) for i in done]
        clash = [p for _, p in targets if os.path.exists(p)]
        if clash and not messagebox.askyesno(
                "Overwrite?", f"{len(clash)} file(s) already exist in that folder. Overwrite?"):
            return
        ok = sum(self._save(i, p) for i, p in targets)
        self._set_status(f"Saved {ok}/{len(targets)} image(s) to {folder}")
        self._refresh()

    def _save(self, it: Item, path: str) -> bool:
        if save_image(it.result['result'], path, dpi=300):
            it.saved_to = path
            self._set_status(f"Saved {path}")
            return True
        messagebox.showerror("Save failed", f"Could not save {path}")
        return False

    # ------------------------------------------------------------- overlays

    def _redraw_overlays(self):
        self.photo_view.redraw_overlay()
        self.output_view.redraw_overlay()

    def _draw_photo_overlay(self, view: ImageView):
        it = self.current
        if it is None:
            return
        if self.show_grid.get():
            H = dist = None
            colour = GRID_AUTO
            if it.corners_edited and corners_valid(it.corners):
                H, colour = grid_corners_homography(it.corners), GRID_MANUAL
            elif not it.corners_edited and it.result is not None and it.result['H'] is not None:
                H, dist = it.result['H'], it.result['distortion']
            if H is not None:
                for line in lattice_polylines():
                    pts = view.to_screen(canvas_to_photo(line, H, dist))
                    view.create_line(*pts.ravel(), fill=colour, width=1, tags='overlay')

        if it.corners is not None:
            pts = view.to_screen(it.corners)
            view.create_polygon(*pts.ravel(), outline=HANDLE, fill='', dash=(4, 3),
                                tags='overlay')
            for k, (x, y) in enumerate(pts):
                c = HANDLE_SEL if k == self.sel_handle else HANDLE
                r = HANDLE_R
                view.create_oval(x - r, y - r, x + r, y + r, outline=c, width=2, tags='overlay')
                view.create_line(x - r - 4, y, x + r + 4, y, fill=c, tags='overlay')
                view.create_line(x, y - r - 4, x, y + r + 4, fill=c, tags='overlay')
            if self.sel_handle is not None:
                self._draw_loupe(view, it, pts[self.sel_handle])

    def _draw_loupe(self, view: ImageView, it: Item, handle_xy):
        """Magnified full-resolution view around the selected corner."""
        cx, cy = it.corners[self.sel_handle]
        half = LOUPE_SRC / 2
        crop = view.full.crop((int(cx - half), int(cy - half),
                               int(cx + half), int(cy + half)))
        self._loupe_img = ImageTk.PhotoImage(crop.resize((LOUPE, LOUPE), Image.BILINEAR))
        # top-left of the view, or top-right if the handle is in the way
        x0 = PAD + 4
        if handle_xy[0] < x0 + LOUPE + 40 and handle_xy[1] < PAD + LOUPE + 40:
            x0 = view.winfo_width() - LOUPE - PAD - 4
        y0 = PAD + 4
        view.create_image(x0, y0, anchor='nw', image=self._loupe_img, tags='overlay')
        view.create_rectangle(x0, y0, x0 + LOUPE, y0 + LOUPE, outline=HANDLE_SEL,
                              width=2, tags='overlay')
        # crosshair at the exact (sub-pixel) handle position
        mx = x0 + (cx - int(cx - half)) * LOUPE / LOUPE_SRC
        my = y0 + (cy - int(cy - half)) * LOUPE / LOUPE_SRC
        view.create_line(x0, my, x0 + LOUPE, my, fill=HANDLE, tags='overlay')
        view.create_line(mx, y0, mx, y0 + LOUPE, fill=HANDLE, tags='overlay')

    def _draw_output_overlay(self, view: ImageView):
        if not self.show_grid.get():
            return
        for k in range(GRID_N + 1):
            a = view.to_screen([[k * CELL, 0], [k * CELL, OUTPUT_SIZE]]).ravel()
            b = view.to_screen([[0, k * CELL], [OUTPUT_SIZE, k * CELL]]).ravel()
            view.create_line(*a, fill=GRID_OUTPUT, tags='overlay')
            view.create_line(*b, fill=GRID_OUTPUT, tags='overlay')

    # -------------------------------------------------------- corner editing

    def _editable(self) -> bool:
        it = self.current
        return it is not None and it.corners is not None and not it.busy

    def _select_handle(self, k: Optional[int]):
        self.sel_handle = k
        self.photo_view.redraw_overlay()

    def _on_press(self, e):
        self.photo_view.focus_set()
        if not self._editable():
            return
        pts = self.photo_view.to_screen(self.current.corners)
        d = np.hypot(pts[:, 0] - e.x, pts[:, 1] - e.y)
        k = int(np.argmin(d))
        self.drag = d[k] <= HANDLE_R + 6
        self._select_handle(k if self.drag else None)

    def _on_drag(self, e):
        if self.drag and self._editable():
            self._move_handle(self.photo_view.to_image(e.x, e.y))

    def _on_release(self, _e):
        self.drag = False

    def _nudge(self, d, step):
        if self.sel_handle is not None and self._editable():
            self._move_handle(self.current.corners[self.sel_handle] + np.multiply(d, step))

    def _move_handle(self, xy):
        it = self.current
        h, w = it.image.shape[:2]
        it.corners[self.sel_handle] = np.clip(xy, 0, [w - 1, h - 1])
        if not it.corners_edited:
            it.corners_edited = True
            self._refresh()
        self.photo_view.redraw_overlay()


def main():
    app = App([p for p in sys.argv[1:] if os.path.isfile(p)])
    app.mainloop()


if __name__ == '__main__':
    main()
