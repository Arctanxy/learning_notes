from tqdm import tqdm
import math
def run():
    command = input().split(' ')
    num1 = int(command[0])
    num2 = int(command[1])
    i = 2
    Prime_list = []
    while len(Prime_list) <= num2:
        prime = True
        for x in range(2,int(math.sqrt(i))):
            if i % x == 0:
                prime = False
                break
        if prime == True:
            Prime_list.append(str(i))
        i += 1
        print(Prime_list)

    print(' '.join(Prime_list[num1-1:]))

if __name__ == "__main__":
    run()