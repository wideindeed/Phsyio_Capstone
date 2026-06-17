; PhysioVision_Setup.iss
; ========================
; Inno Setup 6 script — wraps dist/PhysioVision/ into a
; single Windows installer: PhysioVision_v1_Setup.exe
;
; Download Inno Setup: https://jrsoftware.org/isdl.php
; Open this file in the Inno Setup IDE and click Build → Compile.

#define AppName      "PhysioVision"
#define AppVersion   "1.0.0".
#define AppPublisher "British University in Dubai — Capstone 2026"
#define AppURL       "https://physiovision.app"
#define AppExeName   "PhysioVision.exe"

; ── Point this at your dist/PhysioVision folder ──────────────────
#define SourceDir    "dist\PhysioVision"

[Setup]
AppId={{A3F2B8C1-4D7E-4A2F-9C1D-8E5F3B2A1C0D}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
AllowNoIcons=yes
; Output location and filename
OutputDir=installer_output
OutputBaseFilename=PhysioVision_v1_Setup
; Compression
Compression=lzma2/ultra64
SolidCompression=yes
; Require 64-bit Windows
ArchitecturesInstallIn64BitMode=x64
ArchitecturesAllowed=x64
; Require admin for Program Files installation
PrivilegesRequired=admin
; Show license page (optional — add a LICENSE.txt to enable)
; LicenseFile=LICENSE.txt
; Minimum Windows version: Windows 10
MinVersion=10.0.17763
; Installer appearance
WizardStyle=modern
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\{#AppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon";    Description: "{cm:CreateDesktopIcon}";    GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1

[Files]
; Copy the entire PyInstaller output folder
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}";              Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}";   Filename: "{uninstallexe}"
Name: "{commondesktop}\{#AppName}";     Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
; Offer to launch the app after installation completes
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Clean up any cache files the app creates
Type: filesandordirs; Name: "{app}\__pycache__"
Type: filesandordirs; Name: "{app}\logs"
