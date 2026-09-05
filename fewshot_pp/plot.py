import os
import torch
import plotly.express as px


def plot_sender_head_effect(scores, receiver_list, receiver_input, save_path=None, title=None):
    """
    Layer x head heatmap of the path-patching effect of each candidate sender
    head on the receiver LTHs' `receiver_input` (as produced by
    get_path_patch_head_to_heads).

    scores: tensor/array, shape [n_sender_layers, n_heads] -- results from
        get_path_patch_head_to_heads.
    receiver_list: list[(layer, head)] -- the LTHs being patched into, shown
        in the title.
    receiver_input: "q", "k", or "v" -- which receiver input stream was patched.
    save_path: if given, write an interactive HTML file here.

    Returns the plotly Figure.
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()

    if title is None:
        title = f"Sender-head effect on receiver_input='{receiver_input}' of LTHs {receiver_list}"

    fig = px.imshow(
        scores,
        labels={"x": "Head", "y": "Layer", "color": "Δ L2 norm (relative)"},
        title=title,
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        aspect="auto",
    )
    fig.update_layout(width=800, height=500)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        print(f"saved heatmap to {save_path}")

    return fig
