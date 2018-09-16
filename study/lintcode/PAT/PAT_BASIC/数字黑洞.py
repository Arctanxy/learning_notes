def run():
    a = input()
    if a[0] == a[1] == a[2] == a[3]:
        print("N - N = 0000")
    if a== '6174':
        print("7641 - 1467 = 6174")
    while a != '6174':
        # a = str(a)
        b = int(''.join(sorted(a,reverse=True)))
        c = int(''.join(sorted(a,reverse=False)))
        a = b - c
        a = str(a)
        if len(a) < 4:
            a = ('0' * 4-len(a)) + a
        print(''.join(sorted(a,reverse=True)) +
              ' - ' + ''.join(sorted(a,reverse=False))
              + ' = ' + str(a))


if __name__ == "__main__":
    run()