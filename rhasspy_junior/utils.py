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

import importlib
import typing


def load_class(class_path: str) -> typing.Any:
    last_dot = class_path.rfind(".")
    assert last_dot >= 0

    module_name, class_name = class_path[:last_dot], class_path[last_dot + 1 :]
    module = importlib.import_module(module_name)
    class_object = getattr(module, class_name)

    return class_object
