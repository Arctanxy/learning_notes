N = int(input())
inf = 9999
male_lowest = inf
grade_m = ""
female_highest = 0
grade_f = ""

for i in range(N):
    info = input().split(' ')
    if info[1] == 'M':
        if int(info[3]) < male_lowest:
            male_lowest = int(info[3])
            grade_m = info[0] +' ' + info[2]
    else:
        if int(info[3]) >= female_highest:
            female_highest = int(info[3])
            grade_f = info[0] +' '+ info[2]

if grade_f != "":
    print(grade_f)
else:
    print("Absent")

if grade_m != "":
    print(grade_m)
else:
    print("Absent")

if grade_m == "" or grade_f == "":
    print("NA")
else:
    print(female_highest-male_lowest)