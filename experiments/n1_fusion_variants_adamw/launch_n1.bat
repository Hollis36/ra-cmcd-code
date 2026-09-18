@echo off
cd /d <YOLO11_ROOT>
"<HOME>\AppData\Local\Programs\Python\Python312\python.exe" -u "<PAPER_ROOT>\experiments\n1_fusion_variants_adamw\run_n1.py" > "<PAPER_ROOT>\experiments\n1_fusion_variants_adamw\n1_stdout.log" 2> "<PAPER_ROOT>\experiments\n1_fusion_variants_adamw\n1_stderr.log"
