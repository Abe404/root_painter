mkdir -p target/ubuntu_setup/usr/local/bin
mkdir -p target/ubuntu_setup/DEBIAN
cp src/build/DEBIAN_control target/ubuntu_setup/DEBIAN/control
cp -r target/RootPainter target/ubuntu_setup/usr/local/bin
dpkg-deb --build --root-owner-group target/ubuntu_setup
cp -r target/ubuntu_setup.deb target/RootPainter-ubuntu.deb
