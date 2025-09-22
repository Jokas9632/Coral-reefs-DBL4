import os
import win32com.client
def resolve_shortcut(path):
    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell.CreateShortCut(path)
    return shortcut.Targetpath

lnk_path = r"G:\My Drive\dc4data\coral_bleaching.lnk"
coralbleaching_path = resolve_shortcut(lnk_path)
coralscapes_path = r"G:\My Drive\dc4data\coralscapes"
print("Shortcut points to:", coralbleaching_path)
def walk_and_print(base_path):
    """Recursively walk a directory and print its tree structure."""
    if os.path.exists(base_path):
        print(f"\n📂 Folder found: {base_path}\n")
        for root, dirs, files in os.walk(base_path):
            level = root.replace(base_path, "").count(os.sep)
            indent = " " * 4 * level
            print(f"{indent}[DIR] {os.path.basename(root) if os.path.basename(root) else root}")
            subindent = " " * 4 * (level + 1)
            for f in files:
                print(f"{subindent}[FILE] {f}")
    else:
        print(f"\n❌ Folder not found: {base_path}\n")

# Run for both
for path in [coralscapes_path, coralbleaching_path]:
    walk_and_print(path)




