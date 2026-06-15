#define AppName "Avatar Studio"
#define AppVersion "1.0.0"
#define AppPublisher "Avatar Studio"
#define AppExeName "AvatarStudio.exe"

[Setup]
AppId={{85EA6895-70A4-4DA8-A74D-23C7EC5B975B}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Programs\AvatarStudio
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
MinVersion=10.0.17763
OutputDir=..\dist\AvatarStudio-Installer
OutputBaseFilename=AvatarStudio-Setup
SetupIconFile=..\assets\avatar_studio.ico
UninstallDisplayIcon={app}\{#AppExeName}
WizardStyle=modern
Compression=lzma2/max
SolidCompression=yes
DiskSpanning=yes
DiskSliceSize=2000000000
SlicesPerDisk=5
CloseApplications=yes
RestartApplications=no
SetupLogging=yes
ChangesAssociations=no
UsePreviousAppDir=yes
UsePreviousTasks=yes
VersionInfoVersion={#AppVersion}
VersionInfoProductName={#AppName}
VersionInfoProductVersion={#AppVersion}
VersionInfoCompany={#AppPublisher}
VersionInfoDescription={#AppName} offline installer

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a Desktop shortcut"; GroupDescription: "Shortcuts:"; Flags: checkedonce

[Files]
Source: "..\dist\AvatarStudio\*"; DestDir: "{app}"; Excludes: "_internal\.tv_profile\*;_internal\models\trt_cache\*"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: files; Name: "{app}\*.log"
Type: filesandordirs; Name: "{app}\__pycache__"

[Code]
function InitializeSetup(): Boolean;
begin
  if not IsWin64 then
  begin
    MsgBox(
      'Avatar Studio requires a 64-bit Windows PC.',
      mbError,
      MB_OK
    );
    Result := False;
    exit;
  end;
  Result := True;
end;
