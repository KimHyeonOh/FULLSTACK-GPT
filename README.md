Python 3.10.11

#2.5
#PowerShell에서 가상환경(venv) 실행
Get-ExecutionPolicy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
/env/Scripts/Activate.ps1
pip install -r requirements.txt

-- venv 실행
.\env\Scripts\Activate.ps1
-- venv 탈출
deactivate

#2.6
-- Jupyter Notebook 사용시 Select kernel 해줄 것.

#6 RAG
-- RAG 방식
Stuff, Refine, Map Reduce

#7 Document GPT
streamlit run Home.py
