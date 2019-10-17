def run():
    charactors = input()

    LIM = 1000000007

    count_t,count_p,result = 0,0,0

    for item in charactors:
        if item == 'T':
            count_t += 1

    for char in charactors:
        if char == 'P':
            count_p += 1
        elif char == 'T':
            count_t -= 1
        elif char == 'A':
            result = (result + (count_p * count_t) % LIM) % LIM

    print(result)



if __name__ == "__main__":
    run()