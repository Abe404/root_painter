source ./env/bin/activate
python run_pyinstaller.py
echo "output executable available at dist/main/main"
echo 'list platforms'
ls dist/main/PyQt5/Qt5/plugins/platforms


rm main.spec # temporary file generated as part of the build process.
