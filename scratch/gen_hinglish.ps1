Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
foreach ($v in $synth.GetInstalledVoices()) {
    Write-Host $v.VoiceInfo.Name "---" $v.VoiceInfo.Culture.Name
}

# Generate Hinglish sample:
$synth.SetOutputToWaveFile("scratch\hinglish_sample.wav")
$synth.Speak("Namaste team. Aaj hum discuss karenge Q4 roadmap aur payment gateway launch jo November 10 ko scheduled hai. Michael ne bola ki woh load testing lead karega. Budget approx 25000 dollars approve ho gaya hai. Any questions?")
$synth.Dispose()
Write-Host "Hinglish sample generated!"
