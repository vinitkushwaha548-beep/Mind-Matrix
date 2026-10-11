$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("C:\Users\ASUS\OneDrive\Desktop\MannPulseAI.lnk")
$Shortcut.TargetPath = "python"
$Shortcut.Arguments = "`"C:\Users\ASUS\OneDrive\Desktop\MannPulseAI\python_plotter_mannpulse_auto_interpretation_safe - Copy.py`""
$Shortcut.WorkingDirectory = "C:\Users\ASUS\OneDrive\Desktop\MannPulseAI"
$Shortcut.IconLocation = "C:\Users\ASUS\OneDrive\Desktop\MannPulseAI\app_icon.ico"
$Shortcut.Save()
