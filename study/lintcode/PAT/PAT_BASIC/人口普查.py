def run():
    num = int(input()) # 样本数量
    infomation = {}
    for i in range(num):
        info = input()
        if info == " ":
            continue
        print(info)
        infomation[info[0]] = info[1]
    print(infomation)
    

if __name__ == "__main__":
    run()