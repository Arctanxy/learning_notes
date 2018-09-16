def run():
    CLK_TCK = 100
    c1,c2 = [int(c) for c in input().split(' ')]
    time = c2 - c1
    format_time = round(time/CLK_TCK + 1e-5)

    hour_num = int(format_time/3600)
    hour = convert(hour_num)

    minute_num = int((format_time % 3600) / 60)
    minute = convert(minute_num)

    second_num = format_time % 60
    second = convert(second_num)

    print(hour + ':' + minute + ':' + second)

def convert(t):
    if t < 10:
        return '0' + str(t)
    else:
        return str(t)

if __name__ == "__main__":
    run()