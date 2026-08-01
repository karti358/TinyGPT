param(
    [string]$SvgPath,
    [string]$OutPath
)

$content = [IO.File]::ReadAllText($SvgPath)
$matches = [regex]::Matches($content, 'data:image/png;base64,([A-Za-z0-9+/=]+)')
if ($matches.Count -eq 0) {
    throw "No embedded PNG found in $SvgPath"
}

$largest = $matches | ForEach-Object { $_.Groups[1].Value } | Sort-Object Length -Descending | Select-Object -First 1
$bytes = [Convert]::FromBase64String($largest)
[IO.File]::WriteAllBytes($OutPath, $bytes)
Write-Host "Wrote $OutPath ($($bytes.Length) bytes)"
