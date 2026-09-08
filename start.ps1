param(
    [Parameter(Position=0)]
    [int]$Port = 7860
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Get-LanIPs {
    $ips = @()
    try {
        $ips += Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
            Where-Object {
                $_.IPAddress -notlike "127.*" -and
                $_.PrefixOrigin -ne "WellKnown" -and
                $_.IPAddress -notlike "169.254.*"
            } |
            Select-Object -ExpandProperty IPAddress
    } catch {}
    try {
        $ips += [System.Net.Dns]::GetHostAddresses([System.Net.Dns]::GetHostName()) |
            Where-Object { $_.AddressFamily -eq "InterNetwork" -and $_.IPAddressToString -notlike "127.*" } |
            ForEach-Object { $_.IPAddressToString }
    } catch {}
    $ips | Select-Object -Unique
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "菌落计数 Web 服务" -ForegroundColor Cyan
Write-Host "本机访问:  http://127.0.0.1:$Port"
$lan = Get-LanIPs
if ($lan) {
    Write-Host "局域网访问:"
    foreach ($ip in $lan) {
        Write-Host "  http://${ip}:$Port" -ForegroundColor Green
    }
} else {
    Write-Host "未探测到局域网 IP" -ForegroundColor Yellow
}
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

python webapp.py $Port
