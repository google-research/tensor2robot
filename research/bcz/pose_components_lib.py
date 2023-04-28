# coding=utf-8
# Copyright 2023 The Tensor2Robot Authors.
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

"""Action space definitions for MetaTidy models.
"""

from typing import Text, Tuple

# Name, size, whether it is residual or not, and loss weight.
# This is used to parameterize action labels.
ActionComponent = Tuple[Text, int, bool, float]

# Name, size, whether residual or not.
# This is used to parameterize proprioceptive state inputs.
StateComponent = Tuple[Text, int, bool]

DEFAULT_STATE_COMPONENTS = []
DEFAULT_ACTION_COMPONENTS = [
    ('xyz', 3, True, 100.),
    ('quaternion', 4, False, 10.),
    ('target_close', 1, False, 1.),
]
