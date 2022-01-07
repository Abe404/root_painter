import glob
import os
import shutil
import subprocess
import re
from settings import Settings


def freeze(settings=Settings()):
    if settings.is_mac():
        return freeze_mac(settings)
    if settings.is_linux():
        return freeze_linux(settings)
    if settings.is_windows():
        return freeze_windows(settings)


### Linux ###


def freeze_linux(settings=Settings()):
    pyinstaller(settings, [])

    env_dir = "./env"
    site_packages_dir = os.path.join(env_dir, "lib/python3.6/site-packages")
    build_dir = "./target/RootPainter"
    fix_freeze(build_dir=build_dir, site_packages_dir=site_packages_dir)


### Windows ###


def freeze_windows(settings=Settings()):
    pyinstaller(settings, [])

    env_dir = "./env"
    site_packages_dir = os.path.join(env_dir, "Lib", "site-packages")
    build_dir = "./target/RootPainter"
    fix_freeze(build_dir=build_dir, site_packages_dir=site_packages_dir)


### Mac ###


# TODO: needs rewriting!!!
def get_icons(settings=Settings()):
    """
    Return a list [(size, scale, path)] of available app icons for the current
    platform.
    """
    result = []
    for profile in settings.get_profiles():
        icons_dir = os.path.join("src", "main", "icons", profile)
        for icon_path in glob.glob(f"{icons_dir}/*.png"):
            name = os.path.basename(icon_path)
            match = re.match("(\\d+)(?:@(\\d+)x)?", name)
            if not match:
                print(name)
                raise Exception("Invalid icon name: " + icon_path)
            size, scale = int(match.group(1)), int(match.group(2) or "1")
            result.append((size, scale, icon_path))
    return result


def create_icon_filename(size, scale=1):
    filename = f"icon_{size}x{size}"
    if scale != 1:
        filename += f"@{scale}x"
    filename += ".png"
    return filename


def freeze_mac(settings=Settings()):
    target_dir = os.path.abspath("target")
    freeze_dir = os.path.join(
        target_dir,
    )

    if not os.path.exists(os.path.join(target_dir, "Icon.icns")):
        iconset_path = os.path.join(target_dir, "Icon.iconset")
        os.makedirs(iconset_path, exist_ok=True)

        for size, scale, icon_path in get_icons(settings):
            dest_name = create_icon_filename(size, scale)
            shutil.copy(icon_path, os.path.join(target_dir, "Icon.iconset", dest_name))

        subprocess.check_call(
            ["iconutil", "-c", "icns", os.path.join(target_dir, "Icon.iconset")]
        )

    extra_args = []
    extra_args.extend(["--icon", os.path.join(target_dir, "Icon.icns")])
    extra_args.extend(["-w"])

    pyinstaller(settings, extra_args)

    # Remove pyinstaller
    for unwanted in ("lib", "include", "2to3"):
        remove_if_exists(os.path.join(freeze_dir, "Contents", "MacOS", unwanted))
        remove_if_exists(os.path.join(freeze_dir, "Contents", "Resources", unwanted))

    env_dir = "./env"
    site_packages_dir = os.path.join(env_dir, "lib/python3.6/site-packages")
    build_dir = "./target/RootPainter.app/Contents/MacOS/"
    fix_freeze(build_dir=build_dir, site_packages_dir=site_packages_dir)


def pyinstaller(settings=Settings(), extra_args=[]):
    app_name = "RootPainter"
    target_dir = os.path.abspath("target")

    cmd = []
    cmd.extend(["pyinstaller"])
    cmd.extend(["--name", app_name])
    cmd.extend(["--noupx"])
    cmd.extend(["--log-level", "DEBUG"])
    cmd.extend(["--noconfirm"])
    cmd.extend(extra_args)
    for hidden_import in settings.get_with_default("hidden_imports", []):
        cmd.extend(["--hidden-import", hidden_import])
    cmd.extend(["--distpath", target_dir])
    cmd.extend(["--specpath", os.path.join(target_dir, "PyInstaller")])
    cmd.extend(["--workpath", os.path.join(target_dir, "PyInstaller")])
    cmd.extend(["--debug", "all"])
    cmd.extend([settings.get("main_module")])

    print(" ".join(cmd))

    subprocess.check_call(cmd)


def fix_freeze(build_dir, site_packages_dir):
    """
    If you try to run RootPainter on the command line like so:
    ./target/RootPainter.app/Contents/MacOS/RootPainter
    Then you may receive the following error:
    File "skimage/feature/orb_cy.pyx", line 12, in init skimage.feature.orb_cy
    ModuleNotFoundError: No module named 'skimage.feature._orb_descriptor_positions'

    It seems the built application is missing some crucial files from skimage.
    To copy these accross we will assume you have an environment created with venv (virtual env)
    in the current working directory call 'env'
    """

    # Copy missing orb files
    skimage_dir = os.path.join(site_packages_dir, "skimage")

    orbpy_src = os.path.join(skimage_dir, "feature/_orb_descriptor_positions.py")
    orbpy_target = os.path.join(
        build_dir, "skimage/feature/_orb_descriptor_positions.py"
    )
    shutil.copyfile(orbpy_src, orbpy_target)

    # copy missing orb plugin file
    orbtxt_src = os.path.join(skimage_dir, "feature/orb_descriptor_positions.txt")
    orbtxt_target = os.path.join(
        build_dir, "skimage/io/_plugins/orb_descriptor_positions.txt"
    )
    shutil.copyfile(orbtxt_src, orbtxt_target)

    # Copy missing tiffile plugin
    tif_src = os.path.join(skimage_dir, "io/_plugins/tifffile_plugin.py")
    tif_target = os.path.join(build_dir, "skimage/io/_plugins/tifffile_plugin.py")
    shutil.copyfile(tif_src, tif_target)


def remove_if_exists(_):
    pass


if __name__ == "__main__":
    freeze()
