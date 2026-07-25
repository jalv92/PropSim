; Inno Setup script for PropSim.
;
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" build\PropSim.iss
;
; Per-user install on purpose: PrivilegesRequired=lowest means no UAC prompt and
; no admin rights. The app writes only to the user's own profile, reads the
; user's own NinjaTrader folder, and talks to nothing but 127.0.0.1 -- there is
; nothing here that justifies asking for the machine.

#define AppName    "PropSim"
#define AppVersion "0.1.0"
#define AppPublish "jalv92"
#define AppExe     "PropSim.exe"

[Setup]
AppId={{7F3C1E52-9A44-4C0D-9B21-0E8A6D2C5B11}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublish}
AppSupportURL=https://github.com/jalv92/PropSim
DefaultDirName={localappdata}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
DisableDirPage=auto
PrivilegesRequired=lowest
OutputDir=..\dist_installer
OutputBaseFilename=PropSim-{#AppVersion}-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#AppExe}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; \
  GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; the whole PyInstaller onedir payload
Source: "..\dist\PropSim\*"; DestDir: "{app}"; \
  Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}";              Filename: "{app}\{#AppExe}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}";        Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; \
  Flags: nowait postinstall skipifsilent

[UninstallDelete]
; The rule cache and settings live in the user's profile, not under {app}.
; Deliberately NOT removed on uninstall: a reinstall should not lose the
; accepted disclaimer or a downloaded rule file. Users who want them gone can
; delete %USERPROFILE%\.prop-sim.
Type: filesandordirs; Name: "{app}\_internal\__pycache__"
