def run():
    command = input()
    for i in range(10):
        a = command.count(str(i))
        if a > 0:

            print('%d:%d' % (i,a))

if __name__ == "__main__":
    run()