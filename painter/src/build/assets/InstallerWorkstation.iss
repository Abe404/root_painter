[Setup]
AppName=RootPainter Workstation
AppVersion=0.2.28
AppPublisher=Abraham George Smith
DefaultDirName={autopf}\RootPainter Workstation
DefaultGroupName=RootPainter Workstation
UninstallDisplayIcon={app}\RootPainter.exe
OutputDir=..\..\..\..\dist
OutputBaseFilename=RootPainterWorkstationInstaller
Compression=lzma2
SolidCompression=yes
SetupIconFile=..\..\main\icons\Icon.ico
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Files]
Source: "..\..\..\..\dist\RootPainterWorkstation\*"; DestDir: "{app}"; Flags: recursesubdirs

[Icons]
Name: "{group}\RootPainter Workstation"; Filename: "{app}\RootPainter.exe"
Name: "{autodesktop}\RootPainter Workstation"; Filename: "{app}\RootPainter.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\RootPainter.exe"; Description: "Run RootPainter Workstation"; Flags: nowait postinstall skipifsilent
