@echo off
setlocal

set s1=8-BitNostalgia.mp3
set s2=NeonCircuit.mp3
set s3=TheDungeonoftheLost.mp3
set s4=12Reggae.mp3
set s5=NeonNostalgia_2.mp3
set s6=TheLastLightoftheForest.mp3
set s7=NeonNostalgia.mp3
set s8=TheLastPixel.mp3
set s9=Goblins_Lair.mp3

set flags=--enrich-volume 0.6
set flags2=--dual-chip --enrich-volume 0.6

set input_dir=C:\Users\parallelno\Downloads\
set output_dir=build\04_one_chip

audio2ay preview "%input_dir%\%s1%" "%output_dir%\%s1%" %flags%
audio2ay preview "%input_dir%\%s2%" "%output_dir%\%s2%" %flags%
audio2ay preview "%input_dir%\%s3%" "%output_dir%\%s3%" %flags%
audio2ay preview "%input_dir%\%s4%" "%output_dir%\%s4%" %flags%
audio2ay preview "%input_dir%\%s5%" "%output_dir%\%s5%" %flags%
audio2ay preview "%input_dir%\%s6%" "%output_dir%\%s6%" %flags%
audio2ay preview "%input_dir%\%s7%" "%output_dir%\%s7%" %flags%
audio2ay preview "%input_dir%\%s8%" "%output_dir%\%s8%" %flags%
audio2ay preview "%input_dir%\%s9%" "%output_dir%\%s9%" %flags%

set output_dir=build\04_two_chips
audio2ay preview "%input_dir%\%s1%" "%output_dir%\%s1%" %flags2%
audio2ay preview "%input_dir%\%s2%" "%output_dir%\%s2%" %flags2%
audio2ay preview "%input_dir%\%s3%" "%output_dir%\%s3%" %flags2%
audio2ay preview "%input_dir%\%s4%" "%output_dir%\%s4%" %flags2%
audio2ay preview "%input_dir%\%s5%" "%output_dir%\%s5%" %flags2%
audio2ay preview "%input_dir%\%s6%" "%output_dir%\%s6%" %flags2%
audio2ay preview "%input_dir%\%s7%" "%output_dir%\%s7%" %flags2%
audio2ay preview "%input_dir%\%s8%" "%output_dir%\%s8%" %flags2%
audio2ay preview "%input_dir%\%s9%" "%output_dir%\%s9%" %flags2%