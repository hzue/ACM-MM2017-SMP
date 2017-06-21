def write_csv(pred, ans, path):
    f = open(path, 'w')
    f.write("pred, ans\n")
    for i in range(len(pred)):
        f.write("{}, {}\n".format(pred[i], ans[i]))
    f.close()

