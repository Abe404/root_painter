source ./env/bin/activate
export QT_QPA_PLATFORM_PLUGIN_PATH=env/lib/python3.10/site-packages/PyQt5/Qt5/plugins/platforms
python run_pyinstaller.py
echo "output executable available at dist/main/main"
echo 'list platforms'
ls dist/main/PyQt5/Qt5/plugins/platforms


rm main.spec # temporary file generated as part of the build process.
