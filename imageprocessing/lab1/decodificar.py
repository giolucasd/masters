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
    output_image_path: str = "imagem_saida.png",
    output_message_path: str = "texto_saida.txt",
    bit_layer: int = 0,
):
    output_image = cv2.imread(output_image_path, cv2.IMREAD_COLOR)

    steganographer = TxtInImgSteganography(bit_layer=bit_layer)

    output_message = steganographer.decode(output_image)

    with open(output_message_path, "w") as f:
        f.write(output_message)


if __name__ == "__main__":
    assert (
        len(sys.argv) == 4
    ), "Uso: `python decodificar.py imagem_saida.png plano_bits texto_saida.txt`"
    output_image_path = sys.argv[1]
    bit_layer = sys.argv[2]
    output_message_path = sys.argv[3]

    main(output_image_path, output_message_path, bit_layer)
