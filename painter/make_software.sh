source ./env/bin/activate
python run_pyinstaller.py
echo "output executable available at dist/main/main"
rm RootPainter.spec # temporary file generated as part of the build process.
