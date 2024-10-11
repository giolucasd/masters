import numpy as np
import numpy.typing as npt
from bitstring import BitArray

DTYPE = np.uint8
DTYPE_MAX = np.iinfo(np.uint8).max


class TxtInImgSteganography:
    """
    Steganographer to hide text in images.

    The main algorithm consists in bitwise operating linearized image numpy ndarrays.
    NOTE: Required interface matches numpy and cv2.
    """

    def __init__(self, bit_layer: int = 0):
        self.bit_layer = bit_layer
        self.bit_layer_value = 2**bit_layer
        self.header_dtype = np.uint32
        self.header_size = 32  # 4 bytes is more than enough for 4k images

    def _unpackbits_32_to_8(self, array: npt.NDArray) -> npt.NDArray:
        """Extends np.unpackbits for np.uint32 input array."""
        # TODO: test other byteorders
        return np.unpackbits(array.view(DTYPE)[::-1])

    def _prepare_message(self, message: str, shape: tuple[int, ...]) -> npt.NDArray:
        """
        Prepare message mask for steganography.

        Message mask will be merged with recipient image through bitwise or.
        """
        linear_shape = np.prod(shape)

        assert (
            len(message) * 8 <= linear_shape - self.header_size,
            "Given `message` is too big for the image. Try another combination!",
        )
        # encode message to ascii
        encoded_message = message.encode("ascii", "ignore")
        bitarray_message = (
            np.unpackbits(np.frombuffer(encoded_message, dtype=DTYPE)) << self.bit_layer
        )
        message_size = bitarray_message.shape[0]

        # create header with content size
        bitarray_header = (
            self._unpackbits_32_to_8(np.array([message_size], dtype=self.header_dtype))
            << self.bit_layer
        )

        content_size = message_size + self.header_size

        linear_message_mask = np.hstack(
            (
                bitarray_header,
                bitarray_message,
                np.zeros(shape=(linear_shape - content_size), dtype=DTYPE),
            )
        )
        # For i < content_size:
        #   0,0,0,0,0,0,0,b7 if self.bit_layer == 0
        #   0,0,0,0,0,0,b6,0 if self.bit_layer == 1
        #   ...
        #   b1,0,0,0,0,0,0,0 if self.bit_layer == 7
        # For i >= content_size: 0,0,0,0,0,0,0,0

        return linear_message_mask, content_size

    def _prepare_recipient(
        self, recipient: npt.NDArray, content_size: int
    ) -> npt.NDArray:
        """
        Prepare recipient mask for steganography.

        Recipient mask will be merged with message image through bitwise or.
        """
        linear_shape = np.prod(recipient.shape)
        linear_recipient = np.resize(recipient, linear_shape)
        content_mask = np.hstack(
            (
                np.full(
                    content_size, DTYPE_MAX - self.bit_layer_value, dtype=DTYPE
                ),  # DTYPE_MAX - 1 -> 11111110
                np.full(
                    linear_shape - content_size, DTYPE_MAX, dtype=DTYPE
                ),  # DTYPE_MAX -> 11111111
            )
        )

        linear_recipient_mask = np.where(
            content_mask == DTYPE_MAX - self.bit_layer_value,
            content_mask & linear_recipient,  # b0,b1,b2,b3,b4,b5,b6,0
            linear_recipient,  # b0,b1,b2,b3,b4,b5,b6,b7
        )

        return linear_recipient_mask

    def encode(self, recipient: npt.NDArray, message: str) -> npt.NDArray:
        """Encode message into recipient image using steganography algorithm."""
        linear_message_mask, content_size = self._prepare_message(
            message, recipient.shape
        )
        # For i < content_size: 0,0,0,0,0,0,0,y7
        # For i >= content_size: 0,0,0,0,0,0,0,0

        linear_recipient_mask = self._prepare_recipient(recipient, content_size)
        # For i < content_size: x0,x1,x2,x3,x4,x5,x6,0
        # For i >= content_size: x0,x1,x2,x3,x4,x5,x6,x7

        linear_image = linear_recipient_mask | linear_message_mask
        # For i < content_size:
        # x0,x1,x2,x3,x4,x5,x6,0  | 0,0,0,0,0,0,0,y7 = x0,x1,x2,x3,x4,x5,x6,y7
        # For i >= content_size:
        # x0,x1,x2,x3,x4,x5,x6,x7 | 0,0,0,0,0,0,0,0  = x0,x1,x2,x3,x4,x5,x6,x7

        return np.resize(linear_image, recipient.shape)

    def decode(self, image: npt.NDArray) -> npt.NDArray:
        """Decode message from given image using steganography algorithm."""
        linear_shape = np.prod(image.shape)
        linear_image = np.resize(image, linear_shape)
        bit_mask_1 = np.full(linear_shape, self.bit_layer_value, dtype=DTYPE)
        content_bitarray = linear_image & bit_mask_1
        content_bitarray = content_bitarray >> self.bit_layer

        message_size = BitArray(content_bitarray[: self.header_size]).uint
        message = BitArray(
            content_bitarray[self.header_size : self.header_size + message_size]
        ).bytes.decode("ascii")

        return message
