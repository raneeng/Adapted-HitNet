"""
label_court_notebook.py
------------------------
Interactive court-corner labeling for JupyterLab / notebook environments
where a GUI window (cv2.imshow, used by format_data.py's label_court_corners())
can't open -- e.g. a browser-based JupyterLab terminal with no X server and
no SSH session to forward X11 through in the first place.

This renders inline in the notebook itself instead, using matplotlib's
interactive widget backend, and reuses format_data.py's own write_court_out()
so the output format is guaranteed identical to the GUI-window version.

USAGE -- paste into a notebook cell (not run from a terminal):

    %matplotlib widget
    # (requires ipympl -- if this errors, run: pip install ipympl,
    #  then restart the kernel)

    from label_court_notebook import label_court_corners_notebook
    label_court_corners_notebook("36")

Then, in the cell's output: left-click the image to mark each corner in
order (A top-left, D top-right, G bottom-left, X bottom-right -- watch the
title update after each click), right-click to undo the last one. Once all
4 are marked, run the next cell:

    picker.save()

(the object returned by label_court_corners_notebook() -- keeping the save
step as an explicit, separate call, rather than auto-saving on the 4th
click, so you get a chance to double check the 4 points before committing.)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from format_data import COURT_DIR, write_court_out


LABELS = ["A (top-left)", "D (top-right)", "G (bottom-left)", "X (bottom-right)"]


class CourtCornerPicker:
    """Holds state for one interactive court-corner labeling session.

    Not meant to be constructed directly -- use
    label_court_corners_notebook(), which builds one of these and wires it
    up to a matplotlib figure's click events.

    Attributes:
        match_id: The match id being labeled.
        out_path: Where save() will write the result.
        points:   Clicked (x, y) pixel coordinates so far, in click order
            (should end up as [A, D, G, X] if you click in the prompted order).
    """

    def __init__(self, match_id: str, out_path: Path):
        self.match_id = match_id
        self.out_path = out_path
        self.points: list[tuple[float, float]] = []
        self._fig = None
        self._ax = None
        self._markers = []

    def _on_click(self, event):
        """Matplotlib button_press_event handler -- left-click adds a point, right-click undoes."""
        if event.inaxes != self._ax:
            return
        if event.button == 1 and len(self.points) < 4:  # left click
            self.points.append((event.xdata, event.ydata))
            marker = self._ax.plot(event.xdata, event.ydata, "ro", markersize=8)[0]
            label = self._ax.annotate(
                LABELS[len(self.points) - 1].split()[0],
                (event.xdata, event.ydata),
                color="red", fontsize=12, xytext=(8, 8), textcoords="offset points",
            )
            self._markers.append((marker, label))
        elif event.button == 3 and self.points:  # right click
            self.points.pop()
            marker, label = self._markers.pop()
            marker.remove()
            label.remove()
        self._update_title()
        self._fig.canvas.draw_idle()

    def _update_title(self):
        if len(self.points) < 4:
            self._ax.set_title(f"Match {self.match_id} -- click {LABELS[len(self.points)]}")
        else:
            self._ax.set_title(
                f"Match {self.match_id} -- all 4 marked. "
                f"Run picker.save() to write {self.out_path.name}, "
                f"or right-click to undo the last point."
            )

    def save(self) -> Path:
        """Write the 4 clicked points to the .out file, via write_court_out().

        Returns:
            The path written to.

        Raises:
            ValueError: If fewer than 4 corners have been clicked yet.
        """
        if len(self.points) != 4:
            raise ValueError(
                f"Only {len(self.points)} corner(s) clicked so far -- need "
                f"exactly 4 before saving. Click the remaining corners in "
                f"the figure above, in the order shown in its title."
            )
        write_court_out(self.points, self.out_path)
        print(f"[INFO] Saved {self.out_path}")
        for label, (px, py) in zip(LABELS, self.points):
            print(f"  {label}: ({px:.1f}, {py:.1f})")
        return self.out_path


def label_court_corners_notebook(match_id: str, overwrite: bool = False) -> CourtCornerPicker | Path:
    """Open a match's calibration reference image inline for interactive corner labeling.

    Notebook-native equivalent of format_data.py's label_court_corners() --
    use this one specifically when a GUI window can't open (no X server /
    no SSH session to forward through), which is the common case for a
    browser-based JupyterLab terminal. Requires `%matplotlib widget` to be
    active in the notebook (run that magic in its own cell first; requires
    the ipympl package).

    Args:
        match_id:  The --id value.
        overwrite: If False (default) and {match_id}.out already exists,
            this is skipped and the existing path is returned directly
            rather than opening the picker.

    Returns:
        Either the existing out_path (if skipped because it already
        exists), or a CourtCornerPicker instance -- click on the displayed
        figure to mark corners, then call `.save()` on the returned object
        once all 4 are marked.

    Raises:
        FileNotFoundError: If the calibration reference image doesn't
            exist yet for this match.
    """
    out_path = COURT_DIR / f"{match_id}.out"
    if out_path.is_file() and not overwrite:
        print(f"[SKIP] {out_path} already exists (pass overwrite=True to relabel).")
        return out_path

    ref_path = COURT_DIR / f"{match_id}_calibration_reference.png"
    if not ref_path.is_file():
        raise FileNotFoundError(
            f"Calibration reference image not found: {ref_path}\n"
            f"Run format_data.py for this match first -- it's created "
            f"automatically from the first hit of set 1."
        )

    img = plt.imread(str(ref_path))
    picker = CourtCornerPicker(match_id, out_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)
    picker._fig, picker._ax = fig, ax
    picker._update_title()
    fig.canvas.mpl_connect("button_press_event", picker._on_click)
    plt.show()

    return picker