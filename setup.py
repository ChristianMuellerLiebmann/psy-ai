import os
import subprocess

def is_tool_installed(name):
    try:
        devnull = open(os.devnull)
        subprocess.Popen([name], stdout=devnull, stderr=devnull).communicate()
    except OSError as e:
        if e.errno == os.errno.ENOENT:
            return False
    return True

def install_tool(name, install_command):
    if not is_tool_installed(name):
        os.system(install_command)

install_tool("ffmpeg", "sudo apt-get install ffmpeg")  # Beispiel für Ubuntu

def install_rust():
    # Installieren Sie Rust mit dem Rustup-Installationsskript
    subprocess.run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y", shell=True, check=True)

    # Aktualisieren Sie die PATH-Umgebungsvariable
    os.environ['PATH'] += ":/root/.cargo/bin"

# Überprüfen Sie, ob Rust bereits installiert ist, und installieren Sie es, wenn es nicht ist
if not subprocess.run("rustc --version", shell=True).returncode == 0:
    install_rust()

# Installieren Sie die benötigten Bibliotheken
os.system("pip install whisperx spacy")

# Laden Sie das deutsche Modell für SpaCy herunter
os.system("python -m spacy download de_core_news_sm")
