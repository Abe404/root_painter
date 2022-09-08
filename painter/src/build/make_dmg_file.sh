# run from within painter directory
# bash src/build/make_dmg_file.sh

# Check if create-dmg is installed.
if ! command -v create-dmg &> /dev/null
then
    echo "Requires command create-dmg. To install, run brew install create-dmg"
    exit
fi


# --volname <name>: set volume name (displayed in the Finder sidebar and window title)

# --app-drop-link is important to make it convenient to install.
# --app-drop-link <x> <y>: make a drop link to Applications, at location x, y

create-dmg --volname 'RootPainter' --app-drop-link 10 170 RootPainter.dmg ./dist/RootPainter.app


echo 'output OSX installer to dist/RootPainter.dmg'
