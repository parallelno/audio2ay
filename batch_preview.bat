@echo off
setlocal

set s1=8-BitNostalgia.mp3
set s2=NeonCircuit.mp3
set s3=TheDungeonoftheLost.mp3
set s4=12Reggae.mp3
set s5=NeonNostalgia_2.mp3

set flags=--dual-chip --enrich-volume-b 0.3

set input_dir=C:\Users\parallelno\Downloads\
set output_dir=build\01

audio2ay preview "%input_dir%\%s1%" "%output_dir%\%s1%" %flags%
audio2ay preview "%input_dir%\%s2%" "%output_dir%\%s2%" %flags%
audio2ay preview "%input_dir%\%s3%" "%output_dir%\%s3%" %flags%
audio2ay preview "%input_dir%\%s4%" "%output_dir%\%s4%" %flags%
audio2ay preview "%input_dir%\%s5%" "%output_dir%\%s5%" %flags%
