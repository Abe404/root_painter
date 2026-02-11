; RootPainter Workstation Installer
; Paths are relative to this file's location
!include MUI2.nsh
!include FileFunc.nsh

!define PRODUCT_NAME "RootPainter Workstation"
!define PRODUCT_EXE "RootPainter.exe"
!define DIST_DIR "..\..\..\..\dist\RootPainterWorkstation"
!define ICON_FILE "..\..\..\main\icons\Icon.ico"

!define MUI_ICON "${ICON_FILE}"
!define MUI_UNICON "${ICON_FILE}"

!define VERSION "0.2.28.0"

VIProductVersion "${VERSION}"
VIAddVersionKey "ProductName" "${PRODUCT_NAME}"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "LegalCopyright" "(C) Abraham George Smith"
VIAddVersionKey "FileDescription" "${PRODUCT_NAME}"

;--------------------------------
;Perform Machine-level install, if possible

!define MULTIUSER_EXECUTIONLEVEL Highest
!define MULTIUSER_INSTALLMODE_COMMANDLINE
!include MultiUser.nsh
!include LogicLib.nsh

Function .onInit
  !insertmacro MULTIUSER_INIT
  ${If} $InstDir == ""
      ${If} $MultiUser.InstallMode == "AllUsers"
          StrCpy $InstDir "$PROGRAMFILES\RootPainter Workstation"
      ${Else}
          StrCpy $InstDir "$LOCALAPPDATA\RootPainter Workstation"
      ${EndIf}
  ${EndIf}
FunctionEnd

Function un.onInit
  !insertmacro MULTIUSER_UNINIT
FunctionEnd

;--------------------------------
;General

  Name "${PRODUCT_NAME}"
  OutFile "..\..\..\..\dist\RootPainterWorkstationInstaller.exe"

;--------------------------------
;Interface Settings

  !define MUI_ABORTWARNING

;--------------------------------
;Pages

  !define MUI_WELCOMEPAGE_TEXT "This wizard will guide you through the installation of ${PRODUCT_NAME}.$\r$\n$\r$\n${PRODUCT_NAME} includes both the annotation client and a built-in trainer with GPU support.$\r$\n$\r$\nClick Next to continue."
  !insertmacro MUI_PAGE_WELCOME
  !insertmacro MUI_PAGE_DIRECTORY
  !insertmacro MUI_PAGE_INSTFILES
    !define MUI_FINISHPAGE_NOAUTOCLOSE
    !define MUI_FINISHPAGE_RUN
    !define MUI_FINISHPAGE_RUN_CHECKED
    !define MUI_FINISHPAGE_RUN_TEXT "Run ${PRODUCT_NAME}"
    !define MUI_FINISHPAGE_RUN_FUNCTION "LaunchAsNonAdmin"
  !insertmacro MUI_PAGE_FINISH

  !insertmacro MUI_UNPAGE_CONFIRM
  !insertmacro MUI_UNPAGE_INSTFILES

;--------------------------------
;Languages

  !insertmacro MUI_LANGUAGE "English"

;--------------------------------
;Installer Sections

!define UNINST_KEY \
  "Software\Microsoft\Windows\CurrentVersion\Uninstall\RootPainterWorkstation"
Section
  SetOutPath "$InstDir"
  File /r "${DIST_DIR}\*"
  WriteRegStr SHCTX "Software\RootPainterWorkstation" "" $InstDir
  WriteUninstaller "$InstDir\uninstall.exe"
  CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}.lnk" "$InstDir\${PRODUCT_EXE}"
  WriteRegStr SHCTX "${UNINST_KEY}" "DisplayName" "${PRODUCT_NAME}"
  WriteRegStr SHCTX "${UNINST_KEY}" "UninstallString" \
    "$\"$InstDir\uninstall.exe$\" /$MultiUser.InstallMode"
  WriteRegStr SHCTX "${UNINST_KEY}" "QuietUninstallString" \
    "$\"$InstDir\uninstall.exe$\" /$MultiUser.InstallMode /S"
  WriteRegStr SHCTX "${UNINST_KEY}" "Publisher" "Abraham George Smith"
  WriteRegStr SHCTX "${UNINST_KEY}" "DisplayIcon" "$InstDir\${PRODUCT_EXE}"
  ${GetSize} "$InstDir" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD SHCTX "${UNINST_KEY}" "EstimatedSize" "$0"

SectionEnd

;--------------------------------
;Uninstaller Section

Section "Uninstall"

  RMDir /r "$InstDir"
  Delete "$SMPROGRAMS\${PRODUCT_NAME}.lnk"
  DeleteRegKey /ifempty SHCTX "Software\RootPainterWorkstation"
  DeleteRegKey SHCTX "${UNINST_KEY}"

SectionEnd

Function LaunchAsNonAdmin
  Exec '"$WINDIR\explorer.exe" "$InstDir\${PRODUCT_EXE}"'
FunctionEnd
