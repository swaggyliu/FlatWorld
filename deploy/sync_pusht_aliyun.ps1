# Sync FlatWorld + Push-T weights to the game host. Secrets from deploy/local.env only.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$EnvFile = Join-Path $PSScriptRoot "local.env"
if (-not (Test-Path $EnvFile)) {
  Write-Error "Missing deploy/local.env — copy local.env.example and fill privately."
}

Get-Content $EnvFile | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
  $k, $v = $_ -split '=', 2
  Set-Item -Path "Env:$($k.Trim())" -Value $v.Trim()
}

foreach ($req in @("PUSHT_SSH_HOST", "PUSHT_SSH_USER", "PUSHT_SSH_KEY")) {
  if (-not [string]::IsNullOrWhiteSpace((Get-Item "Env:$req" -ErrorAction SilentlyContinue).Value)) { continue }
  Write-Error "deploy/local.env must set $req"
}

$Key = $env:PUSHT_SSH_KEY
$HostUser = "$($env:PUSHT_SSH_USER)@$($env:PUSHT_SSH_HOST)"
$Remote = if ($env:PUSHT_REMOTE_DIR) { $env:PUSHT_REMOTE_DIR } else { "/home/$($env:PUSHT_SSH_USER)/flatworld" }

if (-not (Test-Path $Key)) { Write-Error "SSH key not found: $Key" }

Write-Host "== pack =="
python (Join-Path $PSScriptRoot "_pack_pusht.py")
$Tar = Join-Path $Root "deploy\_pusht_sync.tgz"
if (-not (Test-Path $Tar)) { Write-Error "pack failed" }

Write-Host "== upload to $HostUser =="
ssh -i $Key -o BatchMode=yes -o IdentitiesOnly=yes $HostUser "mkdir -p $Remote"
scp -i $Key -o BatchMode=yes -o IdentitiesOnly=yes $Tar "${HostUser}:/tmp/_pusht_sync.tgz"
ssh -i $Key -o BatchMode=yes -o IdentitiesOnly=yes $HostUser "mkdir -p $Remote && tar -xzf /tmp/_pusht_sync.tgz -C $Remote && rm /tmp/_pusht_sync.tgz"

Write-Host "== remote reuse venv / restart (optional) =="
# Bash on Linux rejects CRLF (`set: pipefail\r: invalid option name`).
$ReuseSh = Join-Path $PSScriptRoot "remote_reuse_venv.sh"
$ReuseLf = Join-Path $env:TEMP "remote_reuse_venv_lf.sh"
$utf8 = New-Object System.Text.UTF8Encoding $false
[IO.File]::WriteAllText(
  $ReuseLf,
  (([IO.File]::ReadAllText($ReuseSh) -replace "`r`n", "`n") -replace "`r", "`n"),
  $utf8)
scp -i $Key -o BatchMode=yes -o IdentitiesOnly=yes $ReuseLf "${HostUser}:/tmp/remote_reuse_venv.sh"
ssh -i $Key -o BatchMode=yes -o IdentitiesOnly=yes $HostUser "bash /tmp/remote_reuse_venv.sh"

Write-Host "== done. Check health on your TLS host (see local.env PUSHT_TLS_HOST). =="
