# Annotation Generation

`[main]step_V11.1-1_example_GPT-o4-mini_New_GT.py` contains the image/SSI-based annotation workflow and its prompts. The script requests `o4-mini` and writes structured annotation outputs.

Before running, supply the original images, SSI JSON files, and visual-prompt files, then update `vp_folder` and `json_folder` in `main()`. Configure `OPENAI_API_KEY` in the environment. The script imports PyTorch, Pillow, OpenAI, Matplotlib, and tqdm.

This is an experiment script with local path defaults, not a general command-line tool. Run it in a separate working directory: its output handling includes backups of existing output/log files. Input images and annotations are not included.
