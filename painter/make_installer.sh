# from man dpkg-deb
# dpkl-deb --build creates a debian archive from the filesystem tree stroed in binary-directory.
# binary-directory must have a DEBIAN subdirectory, which contains the control
# information files such as the control file itself. These files will be put in the
# binary package's control information area. 

# So what should a control file contain?
# From https://www.debian.org/doc/debian-policy/ch-controlfields#binary-package-control-files-debian-control
# The debian/control file contains the most vital (and version-independent)
# information about the source package and about the binary packages it creates.
# The first paragraph of the control file contains information about the source
# package in general. The subsequent paragraphs each describe a binary package
# that the source tree builds. Each binary package built from this source package
# has a corresponding paragraph, except for any automatically-generated debug
# packages that do not require one.

# Mandatory fields are Package, Version, Architecture, Maintainer and Descript.
# binary pack control file is located in src/build/DEBIAN_control

# the /usr/local/bin folder is required so the deb file unpacks the application into the correct location.
mkdir -p dist/ubuntu_setup/usr/local/bin

# the /usr/share/applications folder is required so the deb file unpacks the an application shortcut into the correct location.
mkdir -p dist/ubuntu_setup/usr/share/applications
cp src/build/RootPainter.desktop dist/ubuntu_setup/usr/share/applications


mkdir dist/ubuntu_setup/DEBIAN
cp -r src/build/DEBIAN_control dist/ubuntu_setup/DEBIAN/control
cp -r dist/RootPainter dist/ubuntu_setup/usr/local/bin/RootPainter
cp -r src/main/icons dist/ubuntu_setup/usr/local/bin/RootPainter/icons

dpkg-deb --build dist/ubuntu_setup
mv dist/ubuntu_setup.deb dist/RootPainter.deb
echo 'output debian package to dist/RootPainter.deb'
