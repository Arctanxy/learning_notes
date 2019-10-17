
def run():
    pose_list = ['B','C','J']
    a_win,b_win,equal = 0,0,0
    N = int(input())
    a_poses = {'J':0,'B':0,'C':0}
    b_poses = {'J':0,'B':0,'C':0}
    for i in range(N):
        pa,pb = input().split(' ')
        # A 赢
        if pose_list.index(pa) - pose_list.index(pb) == -1 or pose_list.index(pa) - pose_list.index(pb) == 2:
            a_win += 1
            a_poses[pa] += 1
        # B 赢
        elif pose_list.index(pa) - pose_list.index(pb) == 1 or pose_list.index(pa) - pose_list.index(pb) == -2:
            b_win += 1
            b_poses[pb] += 1
        elif pa == pb:
            equal += 1

    print(str(a_win)+' '+str(equal)+' '+str(b_win))
    print(str(b_win)+' '+str(equal)+' '+str(a_win))

    a_pose = get_max(a_poses)
    b_pose = get_max(b_poses)
    # print(a_poses,b_poses)
    print(str(a_pose)+' '+str(b_pose))

def get_max(poses):
    a_pose = "B"
    v1 = poses[a_pose]
    for k, v in poses.items():
        if v > v1:
            a_pose = k
        elif v == v1:
            # print(k,a_pose)
            if k < a_pose:
                a_pose = k
            else:
                pass
        else:
            pass
    return a_pose



if __name__ == "__main__":
    run()

