SET s1="8-BitNostalgia.mp3"
SET s2="NeonCircuit.mp3"
SET s3="TheDungeonoftheLost.mp3"
SET s4="12Reggae.mp3"
SET s5="NeonNostalgia_2.mp3"

SET flags=--dual-chip --enrich-volume-b 0.3


SET input_dir="C:\Users\parallelno\Downloads\"
SET output_dir="build\"


audio2ay preview %input_dir%%s1% %output_dir%%s1% %flags%
audio2ay preview %input_dir%%s2% %output_dir%%s2% %flags%
audio2ay preview %input_dir%%s3% %output_dir%%s3% %flags%
audio2ay preview %input_dir%%s4% %output_dir%%s4% %flags%
audio2ay preview %input_dir%%s5% %output_dir%%s5% %flags%