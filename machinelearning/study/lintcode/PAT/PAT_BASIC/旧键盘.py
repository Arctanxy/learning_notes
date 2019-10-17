import re
def run():
    a = input().upper()
    b = input().upper()

    new_list = []
    for c in a:
        if c not in b and c not in new_list:
            new_list.append(c)
    print((''.join(new_list)).upper())

if __name__ == "__main__":
    run()


