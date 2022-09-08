# Copyright (C) 2022 Abraham George Smith
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# run from within painter directory
# bash src/build/make_dmg_file.sh

# Check if create-dmg is installed.
if ! command -v create-dmg &> /dev/null
then
    echo "Requires command create-dmg. To install, run brew install create-dmg"
    exit
fi


# --volname <name>: set volume name (displayed in the Finder sidebar and window title)

# --app-drop-link is important to make it convenient to install.
# --app-drop-link <x> <y>: make a drop link to Applications, at location x, y

create-dmg --volname 'RootPainter' --app-drop-link 10 170 RootPainter.dmg ./dist/RootPainter.app


echo 'output OSX installer to dist/RootPainter.dmg'
