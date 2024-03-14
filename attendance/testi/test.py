from datetime import datetime
time_now = datetime.now()
current_time = time_now.strftime("%H:%M:%S")

print("The current date and time is", current_time)
