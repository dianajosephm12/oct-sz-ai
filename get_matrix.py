arr_str = "0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 1 0 0 1 1 0 1 1 0 0 0 0 0 0 0 0 0 1 1 1 1 0 1 0 0 1 1 1 0 0 0 1 1 0 0 1 1 0 0 1 1 1 1 1 1 1 1 0 1 0 0 1 0 1"
arr = arr_str.split(" ")
control = arr[:36] # 40 for control/fep, 36 for control/chronic
patient = arr[36:]
control_true = 0
control_false = 0
for c in control:
    if c == "0":
        control_true += 1
    else:
        control_false += 1
print("true neg:", control_true, "false pos:", control_false)

patient_true = 0
patient_false = 0
for p in patient:
    if p == "1":
        patient_true += 1
    else:
        patient_false += 1
print("false neg:", patient_false,"true pos:", patient_true)


