import os

def ez_name(x):
    # パックの名前から[a-z0-9]の文字以外を処理して簡単にする
    # ex) "Fraxtil's Beast Beats" は "Fraxtil_sBeastBeats"になる
    x = ''.join(x.strip().split())
    x_clean = []
    for char in x:
        if char.isalnum():
            x_clean.append(char)
        else:
            x_clean.append('_')
    return ''.join(x_clean)

def get_subdirs(root, choose=False):
    # データセット中のパック名(fraxtilならtsunamixとか)を取得する
    # extract.jsonでフラグ"--choose"を指定しているとき、パック名を指定するダイアログを出す
    subdir_names = sorted(list(filter(lambda x: os.path.isdir(os.path.join(root, x)), os.listdir(root))))
    if choose:
        for i, subdir_name in enumerate(subdir_names):
            print('{}: {}'.format(i, subdir_name))
        subdir_idxs = [int(x) for x in input('Which subdir(s)? ').split(',')]
        subdir_names = [subdir_names[i] for i in subdir_idxs]
    return subdir_names