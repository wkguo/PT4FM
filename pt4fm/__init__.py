# Copyright 2026 The PT4FM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PT4FM — Post-Training for Foundation-model VLA (offline RL + SFT).

A thin, config-gated extension on top of RLinf's RECAP pipeline that upgrades
the offline learning signal for pi0.5-class flow-matching VLAs:

* AWR continuous advantage weighting (uses advantage magnitude, not just sign),
* Cal-QL calibration of the value/advantage bootstrap,
* an always-on SFT flow-BC auxiliary loss anchoring the policy to demos.

Only :mod:`pt4fm.losses` is import-light; the model/worker submodules pull in
``rlinf`` and ``openpi`` and should be imported lazily from the entrypoints.
"""

__version__ = "0.1.0"
