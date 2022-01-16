import os
import subprocess
from settings import Settings


def create_installer(settings=Settings()):
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
    # TODO: assertion makensis

    target_dir = os.path.abspath("target")
    installer_path = os.path.join(target_dir, "installer")
    subprocess.check_call(
        ["makensis", "Installer.nsi"], cwd=installer_path, stdout=subprocess.DEVNULL
    )


### Mac ###


def create_installer_mac(settings):

    # TODO: assertion create-dmg
    # TODO: assertion target_dir
    # TODO: assertion icon_filename?

    app_name = settings.get("app_name")
    dmg_name = f"{app_name}.dmg"
    target_dir = os.path.abspath("target")

    dest = os.path.join(target_dir, dmg_name)

    icon_filename = f"{app_name}.app"

    freeze_dir = os.path.join(target_dir, icon_filename)

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


if __name__ == "__main__":
    create_installer()
