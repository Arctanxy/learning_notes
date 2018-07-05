import re
def run():
    cmd = input().split(' ')
    a = cmd[0]
    da = cmd[1]
    b = cmd[2]
    db = cmd[3]
    pa = ''.join(re.findall(da,a))
    pb = ''.join(re.findall(db,b))
    if pa == "":
        pa = 0
    if pb == "":
        pb = 0
    print(int(pa)+int(pb))

if __name__ == "__main__":
    run()