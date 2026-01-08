# SASP project

Dehä fungerar fö MAC. Windows e de typ Command prompt eller git bash iställe för terminal

## Om ni har andra project just nu som använder ssh keys

Kolla om ni har andra ssh keys, skriv i terminalen

MAC: ls -al ~/.ssh

WINDOWS: ls -al %USERPROFILE%/.ssh (kanske?)

## Om int, skapar vi SSH key

skriv dehä i terminalen: 

MAC: ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_aalto -C "din-email@aalto.fi"

WINDOWS: ssh-keygen -t ed25519 -f %USERPROFILE%/.ssh/id_ed25519_aalto -C "din-email@aalto.fi" (kanske?)

Dethär är namnet på din nyckel: "id_ed25519_aalto" du kan ändra ti annat om du vill

Tryck ba enter så skriver du inge lösenord när den frågar fö "passphrase"

## Sätta in i version.aalto.gitlab

Kopiera din nyckel genom att skriva dehär i terminalen:

MAC: cat ~/.ssh/id_ed25519_aalto.pub

WINDOWS: type %USERPROFILE%\.ssh\id_ed25519_aalto.pub (Denhä borde också kopiera rakt)

Kopiera hela texten som börjar med ssh-ed25519 och slutar med din email.

gå ti version.aalto.fi (där vårt project e) och tryck på din profil och "edit profile". Sen på vänster ser du "SSH Keys" tryck där och sen "Add new key". Sen sätter du in den nyckel du nyligen kopierat. Title kan du ändra men lämna resten och tryck add key.


## Clone SASP-projekte in i en folder var du vill spara all kod

MAC & WINDOWS: git clone git@version.aalto.fi:poikela1/sasp-project.git 'folder-name'

eller om du ren gjort en folder kan du söka dig rätt med "cd ~/folder-name" och när du hittat rätt skriva "git clone git@version.aalto.fi:poikela1/sasp-project.git ."

"folder-name" är namnet på foldern du vill spara i

## Testa att den fungerar

Öppna VScode och öppna med den fil du använde i förra steget. Borde se README.md och kanske någå annat som ha blivi tillagt. Kan finnas en "1" ti vänster på "Source control" om så gå dit och pull dom nya ändringarna.


## Adda pytorch

I VScode terminalen

MAC & WINDOWS: 

python -m venv .venv

source .venv/bin/activate
    
pip install torch torchvision torchaudio

Sen:

MAC: cmd+shift+P

Windows: ctrl+shift+p

Välj: 'python: select interpreter' och sen den me (.venv)

borde göra så att den allti aktiverar pytorch. kör 'test.py' Den borde ge 2.9.0 som version

## Adda en kod fil

höger klicka under sasp-project i vscode. Tryck på första "new file" och skriv:
namn.py då blir de en python fil. 


## Om problem me pytorch:


# 1. Delete their old broken project
cd ..
rm -rf SASP-project  ("SASP-project" e folder namne)

# 2. Clone fresh from GitLab
git clone git@version.aalto.fi:polkela1/sasp-project.git SASP-project
cd SASP-project

# 3. Set up their environment
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio



# Ladda ner för att kunna avända mikrofonen under presentationstillfället


# 3. Installera sounddevice och numpy i venv

# Nu, med venv aktiverad, kör:

pip install --upgrade pip
pip install sounddevice numpy


# Nu borde du inte få externally-managed-environment längre, för vi installerar bara inuti din .venv, inte system-Python.

Om du får ett nytt fel som nämner portaudio:
installera biblioteket med Homebrew:

brew install portaudio

och kör sedan:
pip install sounddevice

# 4. Testa att det funkar

Fortfarande i samma terminal (med (.venv) aktiv):

python


# I Python-repl:

import sounddevice as sd
import numpy as np

print(sd.query_devices())


# Om du får en lista med ljudenheter → allt är OK 🎉

Avsluta Python med:

exit()