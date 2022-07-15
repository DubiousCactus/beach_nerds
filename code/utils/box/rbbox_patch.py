#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""


def box_iou_rotated(bboxes1: torch.Tensor,
                    bboxes2: torch.Tensor,
                    mode: str = 'iou',
                    aligned: bool = False,
                    clockwise: bool = True) -> torch.Tensor:
    """Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in
    (x_center, y_center, width, height, angle) format.
    If ``aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.
    .. note::
        The operator assumes:
        1) The positive direction along x axis is left -> right.
        2) The positive direction along y axis is top -> down.
        3) The w border is in parallel with x axis when angle = 0.
        However, there are 2 opposite definitions of the positive angular
        direction, clockwise (CW) and counter-clockwise (CCW). MMCV supports
        both definitions and uses CW by default.
        Please set ``clockwise=False`` if you are using the CCW definition.
        The coordinate system when ``clockwise`` is ``True`` (default)
            .. code-block:: none
                0-------------------> x (0 rad)
                |  A-------------B
                |  |             |
                |  |     box     h
                |  |   angle=0   |
                |  D------w------C
                v
                y (pi/2 rad)
            In such coordination system the rotation matrix is
            .. math::
                \\begin{pmatrix}
                \\cos\\alpha & -\\sin\\alpha \\\\
                \\sin\\alpha & \\cos\\alpha
                \\end{pmatrix}
            The coordinates of the corner point A can be calculated as:
            .. math::
                P_A=
                \\begin{pmatrix} x_A \\\\ y_A\\end{pmatrix}
                =
                \\begin{pmatrix} x_{center} \\\\ y_{center}\\end{pmatrix} +
                \\begin{pmatrix}\\cos\\alpha & -\\sin\\alpha \\\\
                \\sin\\alpha & \\cos\\alpha\\end{pmatrix}
                \\begin{pmatrix} -0.5w \\\\ -0.5h\\end{pmatrix} \\\\
                =
                \\begin{pmatrix} x_{center}-0.5w\\cos\\alpha+0.5h\\sin\\alpha
                \\\\
                y_{center}-0.5w\\sin\\alpha-0.5h\\cos\\alpha\\end{pmatrix}
        The coordinate system when ``clockwise`` is ``False``
            .. code-block:: none
                0-------------------> x (0 rad)
                |  A-------------B
                |  |             |
                |  |     box     h
                |  |   angle=0   |
                |  D------w------C
                v
                y (-pi/2 rad)
            In such coordination system the rotation matrix is
            .. math::
                \\begin{pmatrix}
                \\cos\\alpha & \\sin\\alpha \\\\
                -\\sin\\alpha & \\cos\\alpha
                \\end{pmatrix}
            The coordinates of the corner point A can be calculated as:
            .. math::
                P_A=
                \\begin{pmatrix} x_A \\\\ y_A\\end{pmatrix}
                =
                \\begin{pmatrix} x_{center} \\\\ y_{center}\\end{pmatrix} +
                \\begin{pmatrix}\\cos\\alpha & \\sin\\alpha \\\\
                -\\sin\\alpha & \\cos\\alpha\\end{pmatrix}
                \\begin{pmatrix} -0.5w \\\\ -0.5h\\end{pmatrix} \\\\
                =
                \\begin{pmatrix} x_{center}-0.5w\\cos\\alpha-0.5h\\sin\\alpha
                \\\\
                y_{center}+0.5w\\sin\\alpha-0.5h\\cos\\alpha\\end{pmatrix}
    Args:
        boxes1 (torch.Tensor): rotated bboxes 1. It has shape (N, 5),
            indicating (x, y, w, h, theta) for each row. Note that theta is in
            radian.
        boxes2 (torch.Tensor): rotated bboxes 2. It has shape (M, 5),
            indicating (x, y, w, h, theta) for each row. Note that theta is in
            radian.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).
        clockwise (bool): flag indicating whether the positive angular
            orientation is clockwise. default True.
            `New in version 1.4.3.`
    Returns:
        torch.Tensor: Return the ious betweens boxes. If ``aligned`` is
        ``False``, the shape of ious is (N, M) else (N,).
    """
    assert mode in ['iou', 'iof']
    mode_dict = {'iou': 0, 'iof': 1}
    mode_flag = mode_dict[mode]
    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if aligned:
        ious = bboxes1.new_zeros(rows)
    else:
        ious = bboxes1.new_zeros(rows * cols)
    if not clockwise:
        flip_mat = bboxes1.new_ones(bboxes1.shape[-1])
        flip_mat[-1] = -1
        bboxes1 = bboxes1 * flip_mat
        bboxes2 = bboxes2 * flip_mat
    bboxes1 = bboxes1.contiguous()
    bboxes2 = bboxes2.contiguous()
    ext_module.box_iou_rotated(
        bboxes1, bboxes2, ious, mode_flag=mode_flag, aligned=aligned)
    if not aligned:
        ious = ious.view(rows, cols)
    return ious

def rbbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.
    Args:
        bboxes1 (torch.Tensor): shape (B, m, 5) in <cx, cy, w, h, a> format
            or empty.
        bboxes2 (torch.Tensor): shape (B, n, 5) in <cx, cy, w, h, a> format
            or empty.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
    """
    assert mode in ['iou', 'iof']
    # Either the boxes are empty or the length of boxes's last dimension is 5
    assert (bboxes1.size(-1) == 5 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 5 or bboxes2.size(0) == 0)

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    # resolve `rbbox_overlaps` abnormal when input rbbox is too small.
    clamped_bboxes1 = bboxes1.detach().clone()
    clamped_bboxes2 = bboxes2.detach().clone()
    clamped_bboxes1[:, 2:4].clamp_(min=1e-3)
    clamped_bboxes2[:, 2:4].clamp_(min=1e-3)

    return box_iou_rotated(clamped_bboxes1, clamped_bboxes2, mode, is_aligned)
