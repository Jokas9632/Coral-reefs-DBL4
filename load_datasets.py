import os
import win32com.client
def resolve_shortcut(path):
    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell.CreateShortCut(path)
    return shortcut.Targetpath

data_path = resolve_shortcut(r"G:\\My Drive\\dc4data.lnk")
benthic_path = resolve_shortcut(data_path+r"\\benthic_datasets.lnk")
coralbleaching_path = resolve_shortcut(data_path+r"\\coral_bleaching.lnk")
if not os.path.exists(r"G:\.shortcut-targets-by-id\1v4g4qOrbisBvrpqOxLrYn96nd_gPG_Ge\dc4data\coralscapes"):
     coralscapes_path = resolve_shortcut(data_path+r"\\coralscapes.lnk")
else:
        coralscapes_path = r"G:\.shortcut-targets-by-id\1v4g4qOrbisBvrpqOxLrYn96nd_gPG_Ge\dc4data\coralscapes"
for p in [data_path, benthic_path, coralbleaching_path, coralscapes_path]:
    if os.path.exists(p):
        print(f"Path exists: {p}")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Path does not exist: {p}")
    