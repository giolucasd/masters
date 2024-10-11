# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "bitarray==2.9.2",
#     "bitstring==4.1.4",
#     "numpy==1.26.4",
#     "opencv-python==4.9.0.80",
#     "scikit-image==0.23.2",
#     "scipy==1.13.1",
# ]
# ///
import sys
import cv2
from steganographers.txt_in_img import TxtInImgSteganography


def main(
    input_image_path: str = "imagem_entrada.png",
    input_message_path: str = "texto_entrada.txt",
    output_image_path: str = "imagem_saida.png",
    bit_layer: int = 0,
):
    input_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)

    with open(input_message_path, "r") as f:
        input_message = f.read()

    steganographer = TxtInImgSteganography(bit_layer=bit_layer)

    output_image = steganographer.encode(input_image, input_message)

    cv2.imwrite(output_image_path, output_image)


if __name__ == "__main__":
    assert (
        len(sys.argv) == 5
    ), "Uso: `python codificar.py imagem_entrada.png texto_entrada.txt plano_bits imagem_saida.png`"
    input_image_path = sys.argv[1]
    input_message_path = sys.argv[2]
    bit_layer = sys.argv[3]
    output_image_path = sys.argv[4]

    main(input_image_path, input_message_path, output_image_path, bit_layer)
