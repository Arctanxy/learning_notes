def run():
    command = input().split(' ')
    A = int(command[0])
    B = int(command[1])
    D = int(command[2])

    C = A + B
    shang = C
    result = []

    while shang > 0:
        yu = shang % D
        result.append(yu)
        # print(yu,shang/D)
        shang = int(shang / D)

    # print(result)

    print(''.join([str(i) for i in result[::-1]]))

if __name__ == "__main__":
    run()