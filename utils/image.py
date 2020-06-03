# coding=utf-8
# Copyright 2020 The Tensor2Robot Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for encoding images."""

import io
import numpy as np
from PIL import Image
from PIL import ImageFile


def jpeg_string(image, jpeg_quality = 90):
  """Returns given PIL.Image instance as jpeg string.

  Args:
    image: A PIL image.
    jpeg_quality: The image quality, on a scale from 1 (worst) to 95 (best).

  Returns:
    a jpeg_string.
  """
  # This fix to PIL makes sure that we don't get an error when saving large
  # jpeg files. This is a workaround for a bug in PIL. The value should be
  # substantially larger than the size of the image being saved.
  ImageFile.MAXBLOCK = 640 * 512 * 64

  output_jpeg = io.BytesIO()
  image.save(output_jpeg, 'jpeg', quality=jpeg_quality, optimize=True)
  return output_jpeg.getvalue()


def numpy_to_image_string(image_array, image_format='jpeg',
                          data_type=np.uint8):
  image = Image.fromarray(image_array.astype(data_type))
  output_jpeg = io.BytesIO()
  image.save(output_jpeg, image_format)
  return output_jpeg.getvalue()
