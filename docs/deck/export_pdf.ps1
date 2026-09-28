# Export the deck to PDF with PowerPoint (the challenge form needs both PPTX and PDF).
# Opens PowerPoint briefly. Usage: powershell -File docs/deck/export_pdf.ps1
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$pptx = Join-Path $here "Sahaay_Pitch.pptx"
$pdf = Join-Path $here "Sahaay_Pitch.pdf"
$pp = New-Object -ComObject PowerPoint.Application
$pres = $pp.Presentations.Open($pptx, $true, $false, $false)   # ReadOnly, Untitled=false, WithWindow=false
$pres.SaveAs($pdf, 32)   # ppSaveAsPDF
$pres.Close()
$pp.Quit()
Write-Host "wrote $pdf"
