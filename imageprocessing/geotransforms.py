from typing import List, Sequence

import numpy as np
from manim import (
    BLUE,
    DOWN,
    RED,
    RIGHT,
    UP,
    UR,
    Arrow,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    MathTex,
    Matrix,
    NumberPlane,
    Polygon,
    Scene,
    SurroundingRectangle,
    Transform,
    TransformMatchingTex,
    VGroup,
    Write,
)


def matrix_to_latex(M: np.ndarray, name: str = "H", sigfigs: int = 6) -> str:
    """Convert a numeric matrix to a LaTeX bmatrix string.

    Args:
        M: 2D numpy array containing the matrix values.
        name: Matrix name to show on the left of the equation.
        precision: Number of decimal places for each entry.

    Returns:
        A LaTeX formatted string showing the named matrix.
    """
    rows: List[str] = []
    for row in M:
        formatted_row = " & ".join([f"{val:.{sigfigs}g}" for val in row])
        rows.append(formatted_row)
    body = r" \\ ".join(rows)
    return rf"{name} = \begin{{bmatrix}} {body} \end{{bmatrix}}"


def matrix_bmatrix_latex(M: np.ndarray, sigfigs: int = 6) -> str:
    """Return the LaTeX bmatrix body (no name) for a numeric matrix.

    Args:
        M: 2D numpy array with matrix values.
        precision: Number of decimal places for each entry.

    Returns:
        LaTeX string with the bmatrix for M.
    """
    rows: List[str] = []
    for row in M:
        formatted_row = " & ".join([f"{val:.{sigfigs}g}" for val in row])
        rows.append(formatted_row)
    body = r" \\ ".join(rows)
    return rf"\begin{{bmatrix}} {body} \end{{bmatrix}}"


def apply_homography(
    points: Sequence[Sequence[float]], H: np.ndarray
) -> List[np.ndarray]:
    """Apply a 3x3 homogeneous matrix H to 2D points.

    Args:
        points: Sequence of points, each as [x, y, z] or [x, y, 0].
        H: 3x3 homography matrix.

    Returns:
        List of transformed points as numpy arrays [x', y', 0].
    """
    transformed: List[np.ndarray] = []
    for p in points:
        x, y = float(p[0]) + 3, float(p[1]) + 1  # Shift to true origin
        v = H @ np.array([x, y, 1.0])
        v = v / v[2]
        transformed.append(np.array([v[0] - 3, v[1] - 1, 0.0]))  # Shift back
    return transformed


class IdentityTransform(Scene):
    """Identity transform scene using homogeneous matrix multiplication."""

    def construct(self) -> None:
        axis = NumberPlane(
            x_range=[-2, 8], y_range=[-2, 4], axis_config={"include_numbers": True}
        )
        self.play(Create(axis))

        original_vertices = [[-3, -1, 0], [-3, 1, 0], [0, 1, 0], [0, -1, 0]]
        image = Polygon(*original_vertices, color=BLUE, fill_opacity=0.6)
        self.play(FadeIn(image))
        self.wait(1)

        I = np.eye(3)  # noqa: E741
        matrix_symbolic = MathTex(
            r"I = \begin{bmatrix} 1 & 0 & 0 \\[4pt] 0 & 1 & 0 \\[4pt] 0 & 0 & 1 \end{bmatrix}"
        ).to_corner(UR)
        matrix_numeric = MathTex(matrix_to_latex(I, name="I")).to_corner(UR)

        self.play(Write(matrix_symbolic))
        self.wait(1)
        self.play(TransformMatchingTex(matrix_symbolic, matrix_numeric, run_time=2))
        self.wait(1)

        new_vertices = apply_homography(original_vertices, I)
        warped = Polygon(*new_vertices, color=RED, fill_opacity=0.6)
        self.play(Transform(image, warped), run_time=2)
        self.wait(2)


class TranslationTransform(Scene):
    """Translation using homogeneous matrix multiplication."""

    def construct(self) -> None:
        axis = NumberPlane(
            x_range=[-2, 8], y_range=[-2, 4], axis_config={"include_numbers": True}
        )
        self.play(Create(axis))

        original_vertices = [[-3, -1, 0], [-3, 1, 0], [0, 1, 0], [0, -1, 0]]
        image = Polygon(*original_vertices, color=BLUE, fill_opacity=0.6)
        self.play(FadeIn(image))
        self.wait(1)

        tx, ty = 2.0, 1.0
        T = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])
        matrix_symbolic = MathTex(
            r"T(t_x, t_y) = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix}"
        ).to_corner(UR)
        matrix_numeric = MathTex(matrix_to_latex(T, name="T")).to_corner(UR)

        self.play(Write(matrix_symbolic))
        self.wait(1)
        self.play(TransformMatchingTex(matrix_symbolic, matrix_numeric, run_time=2))
        self.wait(1)

        new_vertices = apply_homography(original_vertices, T)
        warped = Polygon(*new_vertices, color=RED, fill_opacity=0.6)
        self.play(Transform(image, warped), run_time=3)
        self.wait(2)


class ScaleTransform(Scene):
    """Scaling using homogeneous matrix multiplication."""

    def construct(self) -> None:
        axis = NumberPlane(
            x_range=[-2, 8], y_range=[-2, 4], axis_config={"include_numbers": True}
        )
        self.play(Create(axis))

        original_vertices = [[-3, -1, 0], [-3, 1, 0], [0, 1, 0], [0, -1, 0]]
        image = Polygon(*original_vertices, color=BLUE, fill_opacity=0.6)
        self.play(FadeIn(image))
        self.wait(1)

        sx, sy = 2.0, 0.5
        S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])
        matrix_symbolic = MathTex(
            r"S(s_x, s_y) = \begin{bmatrix} s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & 1 \end{bmatrix}"
        ).to_corner(UR)
        matrix_numeric = MathTex(matrix_to_latex(S, name="S")).to_corner(UR)

        self.play(Write(matrix_symbolic))
        self.wait(1)
        self.play(TransformMatchingTex(matrix_symbolic, matrix_numeric, run_time=2))
        self.wait(1)

        new_vertices = apply_homography(original_vertices, S)
        warped = Polygon(*new_vertices, color=RED, fill_opacity=0.6)
        self.play(Transform(image, warped), run_time=4)
        self.wait(2)


class RotationTransform(Scene):
    """Rotation using homogeneous matrix multiplication."""

    def construct(self) -> None:
        axis = NumberPlane(
            x_range=[-2, 8], y_range=[-2, 4], axis_config={"include_numbers": True}
        )
        self.play(Create(axis))

        original_vertices = [[-3, -1, 0], [-3, 1, 0], [0, 1, 0], [0, -1, 0]]
        image = Polygon(*original_vertices, color=BLUE, fill_opacity=0.6)
        self.play(FadeIn(image))
        self.wait(1)

        theta = np.pi / 4
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        matrix_symbolic = MathTex(
            r"R(\theta) = \begin{bmatrix} \cos \theta & -\sin \theta & 0 \\ \sin \theta & \cos \theta & 0 \\ 0 & 0 & 1 \end{bmatrix}"
        ).to_corner(UR)
        matrix_numeric = MathTex(matrix_to_latex(R, name="R")).to_corner(UR)

        self.play(Write(matrix_symbolic))
        self.wait(1)
        self.play(TransformMatchingTex(matrix_symbolic, matrix_numeric, run_time=2))
        self.wait(1)

        new_vertices = apply_homography(original_vertices, R)
        warped = Polygon(*new_vertices, color=RED, fill_opacity=0.6)
        self.play(Transform(image, warped), run_time=4)
        self.wait(2)


class ProjectiveTransform(Scene):
    """Projective (homography) transform using homogeneous matrix multiplication."""

    def construct(self) -> None:
        axis = NumberPlane(
            x_range=[-2, 8], y_range=[-2, 4], axis_config={"include_numbers": True}
        )
        self.play(Create(axis))

        original_vertices = [[-3, -1, 0], [-3, 1, 0], [0, 1, 0], [0, -1, 0]]
        image = Polygon(*original_vertices, color=BLUE, fill_opacity=0.6)
        self.play(FadeIn(image))
        self.wait(1)

        H = np.array([[0.9, 0.1, 0.0], [-0.07, 1.2, -0.3], [-0.08, 0.1, 1.0]])
        matrix_symbolic = MathTex(
            r"H = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\[4pt] h_{21} & h_{22} & h_{23} \\[4pt] h_{31} & h_{32} & h_{33} \end{bmatrix}"
        ).to_corner(UR)
        matrix_numeric = MathTex(matrix_to_latex(H, name="H")).to_corner(UR)

        self.play(Write(matrix_symbolic))
        self.wait(1)
        self.play(TransformMatchingTex(matrix_symbolic, matrix_numeric, run_time=2))
        self.wait(1)

        new_vertices = apply_homography(original_vertices, H)
        warped = Polygon(*new_vertices, color=RED, fill_opacity=0.6)
        self.play(Transform(image, warped), run_time=3)
        self.wait(2)


class HomographyLinearSystem(Scene):
    """Show the linear system A * H = B for 4 matched keypoints (homogeneous coords).

    The scene:
    - plots the numeric source and destination keypoints on the NumberPlane,
    - draws arrows from each source to its corresponding destination,
    - clears the visual plane, and then displays the symbolic matrix equation
      with H symbolic and A/B shown symbolically to illustrate A H = B.
    """

    def construct(self) -> None:
        axis = NumberPlane(
            x_range=[-1, 6], y_range=[-1, 4], axis_config={"include_numbers": True}
        )
        self.play(Create(axis))

        src_points: np.ndarray = np.array(
            [
                [0.6, -0.3, 0.0],
                [0.5, 1.2, 0.0],
                [2.4, 1.6, 0.0],
                [2.3, -0.4, 0.0],
            ]
        )

        dst_points: np.ndarray = np.array(
            [
                [-1.5, -0.5, 0.0],
                [-1.5, 1.5, 0.0],
                [1.5, 1.5, 0.0],
                [1.5, -0.5, 0.0],
            ]
        )

        src_dots = VGroup(
            *[Dot(np.array([p[0], p[1], 0.0]), color=BLUE) for p in src_points]
        )
        dst_dots = VGroup(
            *[Dot(np.array([p[0], p[1], 0.0]), color=RED) for p in dst_points]
        )

        self.play(*[FadeIn(d) for d in src_dots], run_time=1)
        self.play(*[FadeIn(d) for d in dst_dots], run_time=1)

        arrows = VGroup(
            *[
                Arrow(
                    src_dots[i].get_center(),
                    dst_dots[i].get_center(),
                    buff=0.0,
                    stroke_width=2.0,
                    color=UR,
                )
                for i in range(len(src_points))
            ]
        )
        self.play(*[Create(a, run_time=1.0) for a in arrows])
        self.wait(2.0)

        self.play(FadeOut(VGroup(axis, src_dots, dst_dots, arrows)), run_time=1.0)

        # Build matrix objects (Matrix) so we can address entries individually.
        A_entries = [
            [r"x_1", r"y_1", r"1"],
            [r"x_2", r"y_2", r"1"],
            [r"x_3", r"y_3", r"1"],
            [r"x_4", r"y_4", r"1"],
        ]
        H_entries = [
            [r"h_{11}", r"h_{12}", r"h_{13}"],
            [r"h_{21}", r"h_{22}", r"h_{23}"],
            [r"h_{31}", r"h_{32}", r"h_{33}"],
        ]
        B_entries = [
            [r"x_1^\prime", r"y_1^\prime", r"1"],
            [r"x_2^\prime", r"y_2^\prime", r"1"],
            [r"x_3^\prime", r"y_3^\prime", r"1"],
            [r"x_4^\prime", r"y_4^\prime", r"1"],
        ]

        A_mat = Matrix(A_entries).scale(0.9)
        H_mat = Matrix(H_entries).scale(0.9)
        # Use a visible boxed placeholder so MathTex always produces submobjects.
        # We'll replace the boxes with the computed labels during the animation.
        B_mat = Matrix([[r"\boxed{\;}"] * 3 for _ in range(4)]).scale(
            0.9
        )  # start empty

        mul = MathTex(r"\cdot").scale(0.9)
        eq = MathTex(r"=").scale(0.9)

        equation = VGroup(A_mat, mul, H_mat, eq, B_mat).arrange(RIGHT, buff=0.6)
        caption = MathTex(
            r"\text{Solve } A\,H = B \text{ for } H \;(\text{e.g. RANSAC + DLT})"
        ).next_to(equation, DOWN, buff=0.5)

        self.wait(1.0)
        self.play(Write(equation), run_time=1.0)
        self.wait(0.8)
        self.play(Write(caption), run_time=1.0)
        self.wait(0.8)

        # Animate row-by-column multiplication producing B entries incrementally.
        A_entries_list = A_mat.get_entries()
        H_entries_list = H_mat.get_entries()
        B_entries_list = B_mat.get_entries()

        n_rows = 4
        n_cols = 3
        for i in range(n_rows):
            for j in range(n_cols):
                # highlight the i-th row of A
                row_entries = VGroup(*[A_entries_list[i * 3 + k] for k in range(3)])
                row_rect = SurroundingRectangle(row_entries, color=BLUE, buff=0.08)
                # highlight the j-th column of H
                col_entries = VGroup(*[H_entries_list[k * 3 + j] for k in range(3)])
                col_rect = SurroundingRectangle(col_entries, color=RED, buff=0.08)

                self.play(Create(row_rect), Create(col_rect), run_time=0.5)

                # build the symbolic product expression: x_i*h_1j + y_i*h_2j + 1*h_3j
                ai = rf"x_{{{i + 1}}}"
                bi = rf"y_{{{i + 1}}}"
                const = r"1"
                h1 = rf"h_{{1{j + 1}}}"
                h2 = rf"h_{{2{j + 1}}}"
                h3 = rf"h_{{3{j + 1}}}"
                expr = MathTex(rf"{ai}\,{h1} + {bi}\,{h2} + {const}\,{h3}")
                # place expression above the whole equation to avoid overlap
                expr.next_to(equation, UP, buff=0.45)

                self.play(Write(expr), run_time=0.6)
                self.wait(0.3)

                # transform the expression into the entry in B:
                if j == 0:
                    final_label = MathTex(rf"x_{{{i + 1}}}^\prime")
                elif j == 1:
                    final_label = MathTex(rf"y_{{{i + 1}}}^\prime")
                else:
                    final_label = MathTex(r"1")
                final_label.move_to(B_entries_list[i * 3 + j].get_center())

                # replace the placeholder entry in B with the computed label
                self.play(Transform(expr, final_label), run_time=0.5)
                # now set the entry object to the new label (so future transforms target it)
                B_entries_list[i * 3 + j] = final_label
                self.wait(0.2)

                # remove highlights
                self.play(FadeOut(row_rect), FadeOut(col_rect), run_time=0.3)
                self.wait(0.1)

        # final tidy: ensure B_mat shows the symbolic B (like original)
        final_B = Matrix(B_entries).scale(0.9).move_to(B_mat.get_center())
        self.play(Transform(B_mat, final_B), run_time=0.5)
        self.wait(1.0)
