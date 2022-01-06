import math
import os
import platform
import subprocess


def create_installer():
    system_name = platform.system()

    if system_name == "Darwin":
        return create_installer_mac()
    if system_name == "Linux":
        return create_installer_linux()
    if system_name == "Windows":
        return create_installer_windows()


### Linux ###


def create_installer_linux():
    pass


### Windows ###


def create_installer_windows():
    pass


### Mac ###


def create_installer_mac():
    # TODO: Should we create backup / revert to backup on fail / delete backup on success ??
    verbose = True

    app_name = "RootPainter"
    dmg_name = f"{app_name}.dmg"
    target_dir = os.path.abspath("./target")
    dmg_tmp = os.path.join(target_dir, f"rw.{dmg_name}")
    extra_size = 20
    mount_dir = os.path.join("/Volumes", app_name)

    # Clean-up
    cleanup_file_if_exists(os.path.join(target_dir, ".DS_Store"))
    cleanup_file_if_exists(dmg_tmp)
    try:
        # Unmount if already mounted
        detach_image(mount_point=mount_dir)
    except:
        pass

    create_dmg(dest=dmg_tmp, src=target_dir, volume=app_name, verbose=verbose)

    disk_image_size = get_file_size(dmg_tmp)
    disk_image_size += extra_size
    resize_image(image=dmg_tmp, size=disk_image_size, verbose=verbose)

    # Mount image

    attach_image(image=dmg_tmp)

    create_application_link(mount_dir=mount_dir)

    # TODO: Applescript

    # Create a temporary file
    # > APPLESCRIPT=$(mktemp -t createdmg.tmp.XXXXXXXXXX)

    application_clause = 'set position of item "Applications" to {170, 10}'
    positional_clause = f'set position of item "{app_name}.app" to {0, 10}'

    # Get applescript source, replace vars, save to tmp file.
    # > cat applescript_source | sed -e "s/WINX/$WINX/g" -e "s/WINY/$WINY/g" -e "s/WINW/$WINW/g" -e "s/WINH/$WINH/g" -e "s/BACKGROUND_CLAUSE/$BACKGROUND_CLAUSE/g" -e "s/REPOSITION_HIDDEN_FILES_CLAUSE/$REPOSITION_HIDDEN_FILES_CLAUSE/g" -e "s/ICON_SIZE/$ICON_SIZE/g" -e "s/TEXT_SIZE/$TEXT_SIZE/g" | perl -pe  "s/POSITION_CLAUSE/$POSITION_CLAUSE/g" | perl -pe "s/QL_CLAUSE/$QL_CLAUSE/g" | perl -pe "s/APPLICATION_CLAUSE/$APPLICATION_CLAUSE/g" | perl -pe "s/HIDING_CLAUSE/$HIDING_CLAUSE/" >"$APPLESCRIPT"

    # Sleep fix
    # > sleep 2 # pause to workaround occasional "Can’t get disk" (-1728) issues

    # Execute applescript
    # > "/usr/bin/osascript" "${APPLESCRIPT}" "${VOLUME_NAME}" || true

    # No idea what this sleep is for...
    # > sleep 4

    # Cleanup temp file.
    # > rm "$APPLESCRIPT"

    # Fix permissions
    try:
        subprocess.run(f'chmod -Rf go-w "{mount_dir}"')
    except FileNotFoundError:
        # No idea why, but keep getting an error here. Command works when run manually
        # It might have something to do with newer OSX user permission stuff
        print("Could not find file to fix permissions")

    # Bless (open top window on mount)
    subprocess.check_call(["bless", "--folder", mount_dir, "--openfolder", mount_dir])

    detach_image(mount_point=mount_dir)

    # Compress image
    dest = os.path.join(target_dir, dmg_name)
    compress_image(src=dmg_tmp, dest=dest, verbose=verbose)

    cleanup_file_if_exists(dmg_tmp)


def create_dmg(dest, src, volume, verbose=False):
    """
    Create .dmg image from src.
    """
    cmd = [
        "hdiutil",
        "create",
        "-srcfolder",
        src,
        "-volname",
        volume,
        "-fs",
        "HFS+",
        "-fsargs",
        "-c c=64,a=16,e=16",
        "-format",
        "UDRW",
        dest,
    ]
    if verbose:
        cmd.insert(2, "-verbose")
    subprocess.check_call(cmd)


def compress_image(src, dest, verbose=False):
    cmd = [
        "hdiutil",
        "convert",
        src,
        "-format",
        "UDZO",
        "-imagekey",
        "zlib-level=9",
        "-o",
        dest,
    ]
    if verbose:
        cmd.insert(2, "-verbose")
    subprocess.check_call(cmd)


def resize_image(image, size, verbose=False):
    """
    Resize a disk image to specified size.
    """
    cmd = [
        "hdiutil",
        "resize",
        "-size",
        f"{size}m",
        image,
    ]
    if verbose:
        cmd.insert(2, "-verbose")
    subprocess.check_call(cmd)


def attach_image(image):
    """
    Attaches image, returning the device path
    """
    cmd = f"hdiutil attach -readwrite -noverify -noautoopen {image}"
    # Example Output:
    # > /dev/disk7              GUID_partition_scheme
    # > /dev/disk7s1            Apple_HFS                       /Volumes/RootPainter
    output = subprocess.getoutput(cmd)
    # Get the first device path
    for word in output.split():
        if word.startswith("/dev/"):
            return word


def detach_image(mount_point):
    """
    detaches image using device name
    """
    subprocess.run(["hdiutil", "detach", mount_point])


def create_application_link(mount_dir):
    dest = os.path.join(mount_dir, "Applications")
    subprocess.run(["ln", "-s", "/Applications", dest])


def cleanup_file_if_exists(file):
    """
    Remove a file if it already exists.
    """
    try:
        os.remove(file)
    except FileNotFoundError:
        pass


def get_file_size(filepath):
    """
    Get the size of a file in megabytes.
    """
    bytes = os.path.getsize(filepath)
    return bytes_to_megabytes(bytes)


def bytes_to_megabytes(bytes):
    """
    Convert bytes to megabytes, rounding up to nearest integer value.
    """
    return math.ceil(bytes / (1 << 20))


if __name__ == "__main__":
    create_installer()
