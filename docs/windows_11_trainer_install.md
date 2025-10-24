
# Installing the RootPainter trainer (server) on windows 11.

Note: This tutorial assume you have a suitable GPU and CUDA already setup.

1. Open powershell as administrator
2. Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
3. Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
4. Close and reopen powershell as administrator.
5. pyenv install 3.10.11
6. pyenv global 3.10.11
7. pyenv version 
8. python -c "import sys; print(sys.executable)"
9. pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu124
10. pip install root-painter-trainer
11. start-trainer
12. You will be prompted to input the RootPainter Sync directory. You can just type root_painter_sync and press enter.
13. The trainer should now be running and it should tell you your batch size (should be above 0) and if the GPU is available or not (It should say True).
