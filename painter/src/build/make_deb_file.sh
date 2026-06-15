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
# bash src/build/make_deb_file.sh

set -e

# The DEBIAN/control file is GENERATED here rather than copied from a static
# file, so the architecture and version can never drift from the actual build.
# A stale static control file (Architecture: i386 on an amd64 build) previously
# caused dpkg to silently refuse to install the package on clean 64-bit systems.

# Version is read from about.py so it matches the running application
# (same approach as the workstation AppImage build).
VERSION=$(grep -oP 'Version: \K[0-9.]+' src/main/python/about.py)

# Architecture is taken from the build host so the label matches the payload.
ARCH=$(dpkg --print-architecture)

echo "Building RootPainter.deb (version=$VERSION, arch=$ARCH)"

# Lay out the filesystem tree the package will unpack into.
# /usr/local/bin holds the application bundle.
mkdir -p dist/ubuntu_setup/usr/local/bin
cp -r dist/RootPainter dist/ubuntu_setup/usr/local/bin/RootPainter
cp -r src/main/icons dist/ubuntu_setup/usr/local/bin/RootPainter/icons

# /usr/share/applications holds the desktop menu shortcut.
mkdir -p dist/ubuntu_setup/usr/share/applications
cp src/build/RootPainter.desktop dist/ubuntu_setup/usr/share/applications

# Generate the control file. Depends lists the system libraries the bundled
# Qt (xcb) platform plugin loads at runtime; without these declared, apt does
# not pull them in and the GUI fails to start on a minimal install. Most are
# already present on a desktop, but declaring them makes the package robust.
mkdir -p dist/ubuntu_setup/DEBIAN
cat > dist/ubuntu_setup/DEBIAN/control <<EOF
Package: rootpainter
Version: $VERSION
Architecture: $ARCH
Maintainer: Abraham George Smith <ags@di.ku.dk>
Depends: libgl1, libx11-6, libx11-xcb1, libxext6, libdbus-1-3, libfontconfig1, libfreetype6, libglib2.0-0, libxcb1, libxcb-icccm4, libxcb-image0, libxcb-keysyms1, libxcb-randr0, libxcb-render0, libxcb-render-util0, libxcb-shape0, libxcb-shm0, libxcb-sync1, libxcb-xfixes0, libxcb-xinerama0, libxcb-xkb1, libxkbcommon0, libxkbcommon-x11-0
Description: RootPainter enables the rapid training of neural networks for biological image segmentation.
EOF

dpkg-deb --build dist/ubuntu_setup
mv dist/ubuntu_setup.deb dist/RootPainter.deb
echo "output debian package to dist/RootPainter.deb"
