import shutil


def clean():
    """
    Deletes the build target directory.
    """
    shutil.rmtree('./target')


if __name__ == "__main__":
    clean()
