# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/haotian-liu/LLaVA/

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"
DEFAULT_MASK_TOKEN = "<mask>"
DEFAULT_DEPTH_TOKEN = "<depth>"


# STAug Constants

REGION_TOKEN_START = "<|region_token_start|>"
REGION_TOKEN_END = "<|region_token_end|>",
REGION_X0_TOKEN = "<|x_0|>"
REGION_X1_TOKEN = "<|x_1|>"
REGION_X2_TOKEN = "<|x_2|>"
REGION_X3_TOKEN = "<|x_3|>"
REGION_X4_TOKEN = "<|x_4|>"
REGION_X5_TOKEN = "<|x_5|>"
REGION_X6_TOKEN = "<|x_6|>"
REGION_X7_TOKEN = "<|x_7|>"
REGION_Y0_TOKEN = "<|y_0|>"
REGION_Y1_TOKEN = "<|y_1|>"
REGION_Y2_TOKEN = "<|y_2|>"
REGION_Y3_TOKEN = "<|y_3|>"
REGION_Y4_TOKEN = "<|y_4|>"
REGION_Y5_TOKEN = "<|y_5|>"
REGION_Y6_TOKEN = "<|y_6|>"
REGION_Y7_TOKEN = "<|y_7|>"
ENHANCE_IMG_PLACEHOLDER = "<|enhance_pad|>" 