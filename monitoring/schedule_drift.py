import time
import schedule
import subprocess

def run_drift_check():
    print("🚀 Lancement du rapport Evidently...")
    subprocess.run(["python3", "monitoring/run_evidently.py"])

# Planification : toutes les minutes (pour tester)
schedule.every(1).minutes.do(run_drift_check)

print("✅ Scheduler lancé (CTRL+C pour arrêter)")
while True:
    schedule.run_pending()
    time.sleep(1)


