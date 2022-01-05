from clean import clean
from freeze import freeze
from install_fixes import fix_app
from installer import create_installer


def main():
    clean()
    freeze()
    fix_app()
    create_installer()


if __name__ == "__main__":
    main()
