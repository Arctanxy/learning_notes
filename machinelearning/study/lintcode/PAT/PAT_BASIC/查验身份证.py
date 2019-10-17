def run():

    num = int(input())
    id_list = []
    while len(id_list) < 4:
        try:
            c = input()
        except:
            pass
        if len(c) == 18:
            id_list.append(c)
    all_passed = True
    for id_num in id_list:
        all_passed = check(id_num,all_passed)
    if all_passed:
        print("All passed")

def check(id_num,all_passed):
    z_dict = {
        0:1,1:0,2:'X',3:9,4:8,5:7,6:6,7:5,8:4,9:3,10:2
    }
    ks = [7,9,10,5,8,4,2,1,6,3,7,9,10,5,8,4,2]
    sum = 0
    for i,n in enumerate(id_num[:17]):
        try:
            m = int(n)
        except:
            print(id_num)
            all_passed = False
            break

        sum += m * ks[i]
    result = sum % 11
    result = z_dict[result]
    if str(result) == id_num[17]:
        pass
    else:
        print(id_num)
        all_passed = False
    return all_passed

if __name__ == '__main__':
    run()
