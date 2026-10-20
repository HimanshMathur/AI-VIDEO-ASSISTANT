Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile("scratch\speech_sample.wav")
$synth.Speak("Hello team, welcome to the weekly engineering review. Sarah announced that the new payment system is ready for launch next Tuesday. Michael will lead load testing and performance benchmarks. The leadership team decided to adopt local AI processing to guarantee privacy.")
$synth.Dispose()
Write-Host "Audio generated successfully!"
