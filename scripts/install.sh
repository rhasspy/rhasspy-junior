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
# Creates a virtual environment and installs requirements.
#
set -eo pipefail

venv_dir="$1"

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
base_dir="$(realpath "${this_dir}/..")"

if [ -z "${venv_dir}" ]; then
    venv_dir="${base_dir}/.venv"
fi

python3 -m venv "${venv_dir}"
source "${venv_dir}/bin/activate"

pip3 install --upgrade pip
pip3 install -r "${base_dir}/requirements.txt"
