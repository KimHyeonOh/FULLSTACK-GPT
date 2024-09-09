Python 3.10.11

#PowerShell에서 가상환경(venv) 실행
Get-ExecutionPolicy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
/env/Scripts/Activate.ps1

pip install -r requirements.txt

-- source env/bin/activate (Mac일 경우 venv)

-- venv 실행
.\env\Scripts\Activate.ps1

-- venv 탈출
deactivate
