import os

dataPath = '../../dataset/paper_dataset/20s_path_gt.csv'
dataRoot = '../../dataset/paper_dataset'
#savePath = './result/test_10s_1.csv'
adjust = "24Hz_limit_bpm_144_length"
savePath = "./result/"+adjust+"_10s.csv"
for i in range(1, 4):
    dataPath = list(dataPath)
    dataPath[-15] = str(i)
    dataPath = ''.join(dataPath)

    savePath = list(savePath)
    savePath[-7] = str(i)
    savePath = ''.join(savePath)
    #savePath[-10] = str(i)
    #os.system('python ./record_result.py -d ', dataPath, '-r ', dataRoot, ' --savePath', savePath, '-s ', str(i*10))
    comment = 'python ./record_result.py -d ' + dataPath + ' -r ' + dataRoot + ' --savePath ' + savePath + ' -s ' + str(i*10)
    os.system(comment)
    #print(comment)