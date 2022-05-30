#!/usr/bin/env bash
# Copyright 2022 Michael Hansen
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# -----------------------------------------------------------------------------
#
# Runs Rhasspy Junior.
#
set -eo pipefail

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
base_dir="$(realpath "${this_dir}/..")"

if [ -d "${base_dir}/.venv" ]; then
    source "${base_dir}/.venv/bin/activate"
fi

PYTHONPATH="${base_dir}:${PYTHONPATH}" \
    python3 -m rhasspy_junior "$@"
