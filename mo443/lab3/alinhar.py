# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy==1.26.4",
#     "opencv-python==4.9.0.80",
#     "scikit-image==0.23.2",
#     "scipy==1.13.1",
# ]
# ///
import sys
import cv2
from aligners import HoughAligner, HorizontalProjectionAligner

def get_aligner(mode: str):
    if mode == "hough":
        return HoughAligner()
    return HorizontalProjectionAligner()


def main(
    input_image_path: str = "inputs/neg_28.png",
    mode: str = "horizontal_projection",
    output_image_path: str = "outputs/neg_28.png",
):
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    print(input_image.dtype, input_image.shape)

    aligner = get_aligner(mode)

    output_image = aligner.align(input_image)

    print(output_image.dtype, output_image.shape)

    cv2.imwrite(output_image_path, output_image)

    return 0


if __name__ == "__main__":
    assert (
        len(sys.argv) == 4
    ), "Uso: `python alinhar.py imagem_entrada.png modo imagem_saida.png`"
    input_image_path = sys.argv[1]
    mode = sys.argv[2]
    output_image_path = sys.argv[3]

    main(input_image_path, mode, output_image_path)
