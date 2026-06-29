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

"""PT4FM embodiment integrations (openpi data transforms + config registration)."""


def register_all(repo_id=None):
    """Register every PT4FM-provided openpi config into RLinf's registry."""
    from pt4fm.integrations.industrial_arm import register_industrial_arm_configs

    register_industrial_arm_configs(repo_id=repo_id)
