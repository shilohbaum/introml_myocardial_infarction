import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# Train
# 113pMI is broken

# Test
# 61 is odd

# Load 20 samples
data_dir = Path("./Data/test_samples/test_samples")
# data_dir = Path("./Data/training_data/mixed_samples")
filepaths = sorted(data_dir.glob("*.npy"))[80:]

samples = [np.load(fp) for fp in filepaths]

assert len(samples) == 20, "Expected 20 samples"

# Use the shortest animation length
n_frames = min(len(s) for s in samples)

# Global axis limits across all samples and frames
xyz_min = np.min(
    [s[:, :, :3].min(axis=(0, 1)) for s in samples],
    axis=0
)
xyz_max = np.max(
    [s[:, :, :3].max(axis=(0, 1)) for s in samples],
    axis=0
)

# Global colour limits
cmin = min(s[:, :, 3].min() for s in samples)
cmax = max(s[:, :, 3].max() for s in samples)

# Create 4x5 grid
fig = make_subplots(
    rows=4,
    cols=5,
    specs=[[{"type": "scene"}] * 5 for _ in range(4)],
    subplot_titles=[fp.stem for fp in filepaths],
    horizontal_spacing=0.01,
    vertical_spacing=0.04,
)

# Initial traces
for j, sample in enumerate(samples):
    row = j // 5 + 1
    col = j % 5 + 1

    fig.add_trace(
        go.Scatter3d(
            x=sample[0, :, 0],
            y=sample[0, :, 1],
            z=sample[0, :, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=sample[0, :, 3],
                colorscale="Viridis",
                cmin=cmin,
                cmax=cmax,
                showscale=False,
            ),
            showlegend=False,
        ),
        row=row,
        col=col,
    )

# Animation frames (update all 20 traces together)
frames = []

for i in range(n_frames):
    frame_data = []

    for sample in samples:
        frame_data.append(
            go.Scatter3d(
                x=sample[i, :, 0],
                y=sample[i, :, 1],
                z=sample[i, :, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=sample[i, :, 3],
                    colorscale="Viridis",
                    cmin=cmin,
                    cmax=cmax,
                    showscale=False,
                ),
                showlegend=False,
            )
        )

    frames.append(
        go.Frame(
            data=frame_data,
            traces=list(range(20)),
            name=str(i),
        )
    )

fig.frames = frames

# Fix axis ranges for all 20 scenes
for j in range(20):
    scene_name = "scene" if j == 0 else f"scene{j+1}"

    fig.layout[scene_name].update(
        xaxis=dict(
            range=[xyz_min[0], xyz_max[0]],
            autorange=False,
            visible=False,
        ),
        yaxis=dict(
            range=[xyz_min[1], xyz_max[1]],
            autorange=False,
            visible=False,
        ),
        zaxis=dict(
            range=[xyz_min[2], xyz_max[2]],
            autorange=False,
            visible=False,
        ),
        aspectmode="data",
    )

# Layout and animation controls
fig.update_layout(
    width=1600,
    height=1300,
    margin=dict(l=10, r=10, t=60, b=80),

    sliders=[dict(
        active=0,
        steps=[
            dict(
                method="animate",
                args=[
                    [str(i)],
                    dict(
                        mode="immediate",
                        frame=dict(duration=0, redraw=True),
                        transition=dict(duration=0),
                    ),
                ],
                label=str(i),
            )
            for i in range(n_frames)
        ],
        currentvalue=dict(prefix="Frame: "),
    )],

    updatemenus=[dict(
        type="buttons",
        showactive=False,
        x=0,
        y=-0.04,
        buttons=[
            dict(
                label="Play",
                method="animate",
                args=[
                    None,
                    dict(
                        frame=dict(duration=100, redraw=True),
                        transition=dict(duration=0),
                        fromcurrent=True,
                    ),
                ],
            ),
            dict(
                label="Pause",
                method="animate",
                args=[
                    [None],
                    dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                    ),
                ],
            ),
        ],
    )],
)

fig.show()

