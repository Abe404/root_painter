import os
import subprocess
import shutil
from settings import Settings


def create_installer(settings):
    if settings.is_mac():
        return create_installer_mac(settings)
    if settings.is_linux():
        return create_installer_linux(settings)
    if settings.is_windows():
        return create_installer_windows(settings)


### Linux ###


def create_installer_linux(_):
    pass


### Windows ###


def create_installer_windows(_):
    target_dir = os.path.abspath("target")
    installer_path = os.path.join(target_dir, "PyInstaller")

    check_cmd_exists(
        "makensis",
        "Requires command `makensis`. Install from https://nsis.sourceforge.io/Main_Page",
    )
    check_dir_exists(
        target_dir, "Target directory is missing. Have you run freeze command?"
    )
    check_dir_exists(
        installer_path, "Installer directory is missing. Have you run freeze command?"
    )

    shutil.copyfile(
        os.path.join("src", "build", "assets", "Installer.nsi"),
        os.path.join(installer_path, "Installer.nsi"),
    )

    subprocess.check_call(
        ["makensis", "Installer.nsi"], cwd=installer_path, stdout=subprocess.DEVNULL
    )


### Mac ###


def create_installer_mac(settings):

    app_name = settings.get("app_name")
    dmg_name = f"{app_name}.dmg"
    target_dir = os.path.abspath("target")
    dest = os.path.join(target_dir, dmg_name)
    icon_filename = f"{app_name}.app"
    freeze_dir = os.path.join(target_dir, icon_filename)

    check_cmd_exists(
        "create-dmg",
        "Requires command `create-dmg`. To install, run `brew install create-dmg`",
    )
    check_dir_exists(
        target_dir, "Target directory is missing. Have you run freeze command?"
    )
    check_dir_exists(
        freeze_dir, "Application directory missing. Have you run freeze command?"
    )

    cmd = [
        "create-dmg",
        "--no-internet-enable",
        "--hdiutil-verbose",
        "--volname",
        app_name,
        "--app-drop-link",
        "170",
        "10",
        "--icon",
        icon_filename,
        "0",
        "10",
        dest,
        freeze_dir,
    ]

    subprocess.check_call(cmd)


def check_cmd_exists(cmd, msg):
    if shutil.which(cmd) is None:
        raise FileNotFoundError(msg)


def check_dir_exists(dir, msg):
    if not os.path.isdir(dir):
        raise FileNotFoundError(msg)


if __name__ == "__main__":
    create_installer(settings=Settings())
